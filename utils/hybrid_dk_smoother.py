import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax, random, vmap
from jax.typing import ArrayLike
from typing import Tuple, Optional, Union, Sequence, Dict, Any
import time # Keep time for profiling if needed

# Force JAX to use CPU - helps with control flow (Optional, can remove for GPU)
# jax.config.update("jax_platforms", "cpu")

# Enable double precision (should match other JAX code)
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

# Reduced jitter value for numerical stability (should match Kalman filter)
_KF_JITTER_DK = 1e-8

# Reuse Kalman filter logic (copy or import specific methods)
# Since the prompt says these are "working perfectly", let's assume
# the necessary filter/smoother logic is included or adapted within this class.
# Based on the provided code structure, it seems the _filter_internal and
# _rts_smoother_backend methods ARE intended to be internal copies/adaptations.

class HybridDKSimulationSmoother:
    def __init__(self, T: jax.Array, R: jax.Array, C: jax.Array, H_full: jax.Array,
                 init_x_mean: jax.Array, init_P_cov: jax.Array):
        """
        Hybrid Durbin-Koopman Simulation Smoother that selects the most efficient
        algorithm based on state dimension size. Handles dense observations.

        Args:
            T: State transition matrix (n_state, n_state)
            R: Shock impact matrix (n_state, n_shocks) - R @ R.T is state noise cov
            C: Observation matrix (n_obs_full, n_state) - maps state to full observation space
            H_full: Full observation noise covariance matrix (n_obs_full, n_obs_full)
            init_x_mean: Initial state mean (n_state,)
            init_P_cov: Initial state covariance (n_state, n_state)
        """
        # Store inputs with correct dtype
        desired_dtype = _DEFAULT_DTYPE
        self.T = jnp.asarray(T, dtype=desired_dtype)
        self.R = jnp.asarray(R, dtype=desired_dtype)
        self.C_full = jnp.asarray(C, dtype=desired_dtype) # C matrix for full observation space
        self.H_full = jnp.asarray(H_full, dtype=desired_dtype) # H matrix for full observation space
        self.init_x = jnp.asarray(init_x_mean, dtype=desired_dtype)
        self.init_P = jnp.asarray(init_P_cov, dtype=desired_dtype)

        # Extract dimensions
        self.n_state = self.T.shape[0]
        self.n_obs_full = self.C_full.shape[0] # Dimension of the full observation vector
        self.n_shocks = self.R.shape[1] if self.R.ndim == 2 and self.R.shape[1] > 0 else 0

        # Determine if we should use small or large state implementation
        # Use a threshold of 100 state variables based on performance tests
        # Note: The 'large' implementation *is* the 'small' one in the code below.
        # The original PyTensor code had different algorithms. Here, the flag only
        # affects the vmap strategy in run_smoother_draws.
        self.use_small_implementation = self.n_state <= 100 # This might need tuning

        # Initialize helper values
        self.I_s = jnp.eye(self.n_state, dtype=desired_dtype)
        # Identity matrix for the FULL observation space
        self.I_obs_full = jnp.eye(self.n_obs_full, dtype=desired_dtype) if self.n_obs_full > 0 else jnp.empty((0,0), dtype=desired_dtype)

        # Pre-compute state covariance (Q = R @ R.T)
        if self.n_shocks > 0:
            self.state_cov_Q = self.R @ self.R.T
        else:
            self.state_cov_Q = jnp.zeros((self.n_state, self.n_state), dtype=desired_dtype)

        # Setup observation noise simulation - used by _draw_simulation_*
        if self.n_obs_full > 0:
            H_reg = self.H_full + _KF_JITTER_DK * self.I_obs_full
            try:
                # Cholesky of H_full for simulation (used by _simulate_obs_noise_chol)
                self.L_H_full = jnp.linalg.cholesky(H_reg)
                self._simulate_obs_noise = self._simulate_obs_noise_chol
            except:
                # Fallback for simulation if Cholesky fails (used by _simulate_obs_noise_mvn)
                self.H_stable_full = H_reg + _KF_JITTER_DK * 10 * self.I_obs_full # Add more jitter for MVN stability?
                self._simulate_obs_noise = self._simulate_obs_noise_mvn
        else:
            self._simulate_obs_noise = self._simulate_obs_noise_empty

        # Pre-compute initial state decomposition for simulation
        try:
            init_P_reg = self.init_P + _KF_JITTER_DK * self.I_s
            # Cholesky of init_P for initial state simulation
            self.L0 = jnp.linalg.cholesky(init_P_reg)
            self.init_chol_success = True
        except:
            self.L0 = None
            self.init_chol_success = False


    # --- Observation noise simulation methods ---
    def _simulate_obs_noise_empty(self, key, shape):
        """Dummy simulation for zero observation dimension."""
        return jnp.empty(tuple(shape) + (0,), dtype=self.H_full.dtype)

    def _simulate_obs_noise_chol(self, key, shape):
        """Simulate observation noise using Cholesky decomposition of H_full."""
        # shape should be (num_steps,) or similar batch dimensions
        # Result shape should be shape + (n_obs_full,)
        z_eta = random.normal(key, tuple(shape) + (self.n_obs_full,), dtype=self.H_full.dtype)
        return z_eta @ self.L_H_full.T # Simulates N(0, L_H @ L_H.T = H_full)

    def _simulate_obs_noise_mvn(self, key, shape):
        """Simulate observation noise using multivariate_normal if Cholesky fails."""
        mvn_shape = tuple(shape) if len(shape) > 0 else () # Shape for mvn function
        try:
            return random.multivariate_normal(
                key,
                jnp.zeros((self.n_obs_full,), dtype=self.H_full.dtype),
                self.H_stable_full, # Use the regularized H
                shape=mvn_shape,
                dtype=self.H_full.dtype
            )
        except:
            # If MVN also fails (highly unlikely with regularization), return zeros
            print("Warning: simulate_obs_noise_mvn failed. Returning zeros for noise.")
            return jnp.zeros(tuple(shape) + (self.n_obs_full,), dtype=self.H_full.dtype)


    # --- Filter for original data (or simulated data) ---
    # This filter implementation assumes observations are DENSE (no NaNs within the matrix)
    # or that any NaNs are handled by pre-processing or the specific use case.
    # It does NOT use the static_* logic from Kalman_filter_jax.filter
    # It's a simpler filter designed for dense data inputs like generated simulated data,
    # or original data if it contains no NaNs.
    # If original_ys has NaNs, you would need to modify this filter
    # or impute/handle them *before* passing to this smoother.
    # Let's assume for now the original_ys passed to smooth_and_simulate
    # are either dense or NaNs are handled externally for the *initial* filter pass.
    # The draw_simulation methods generate dense data, so this filter is fine for them.
    # Reworking this filter to handle time-varying NaNs is a separate, complex task.

    def _filter_internal(self, ys: jax.Array) -> Dict[str, jax.Array]:
        """
        Internal Kalman filter implementation for dense observations.
        Simplified from Kalman_filter_jax.filter by assuming `ys` is dense
        and doesn't require time-varying NaN handling or static observation indices.
        """
        ys_arr = jnp.asarray(ys, dtype=_DEFAULT_DTYPE)
        T_mat, I_s = self.T, self.I_s
        state_cov_Q_internal = self.state_cov_Q # This is Q (R @ R.T)
        C_dense = self.C_full # Use the full C matrix
        H_dense = self.H_full # Use the full H matrix
        I_obs_dense = self.I_obs_full # Use the full I_obs

        kf_jitter = _KF_JITTER_DK
        MAX_STATE_VALUE = 1e6
        T_steps = ys_arr.shape[0]
        n_obs_dense = self.n_obs_full # Dimension of the dense observation vector

        # Handle empty time series case
        if T_steps == 0:
            obs_dim = n_obs_dense if n_obs_dense > 0 else 0
            return {'x_pred': jnp.empty((0,self.n_state),dtype=I_s.dtype),
                   'P_pred': jnp.empty((0,self.n_state,self.n_state),dtype=I_s.dtype),
                   'x_filt': jnp.empty((0,self.n_state),dtype=I_s.dtype),
                   'P_filt': jnp.empty((0,self.n_state,self.n_state),dtype=I_s.dtype),
                   'innovations': jnp.empty((0,obs_dim),dtype=ys_arr.dtype),
                   'innovation_cov': jnp.empty((0,obs_dim,obs_dim),dtype=ys_arr.dtype),
                   'log_likelihood_contributions': jnp.empty((0,),dtype=I_s.dtype)}

        def step_dense_data(carry, y_t_slice):
            # carry: (x_prev_filt, P_prev_filt)
            # y_t_slice: observation vector at time t, shape (n_obs_full,)

            x_prev_filt, P_prev_filt = carry

            # --- Prediction Step ---
            x_pred_t = T_mat @ x_prev_filt
            x_pred_t = jnp.clip(x_pred_t,-MAX_STATE_VALUE,MAX_STATE_VALUE)

            # Ensure P_prev_filt is symmetric and regularized before prediction
            P_prev_filt_sym = (P_prev_filt+P_prev_filt.T)/2.0
            P_prev_filt_reg = P_prev_filt_sym+kf_jitter*I_s

            P_pred_t = T_mat @ P_prev_filt_reg @ T_mat.T + state_cov_Q_internal
            # Ensure P_pred_t is symmetric and regularized after prediction
            P_pred_t=(P_pred_t+P_pred_t.T)/2.0
            P_pred_t=P_pred_t+kf_jitter*I_s

            # --- Update Step ---
            # For dense filter, assume y_t_slice is the observation vector
            y_obs_t = y_t_slice
            y_pred_obs = C_dense @ x_pred_t
            v_obs = y_obs_t - y_pred_obs # Innovation

            # Initialize K, S, log-likelihood contribution with proper shapes
            # Fixed: Ensure all return values have consistent shapes regardless of n_obs_dense
            if n_obs_dense > 0:
                K_default = jnp.zeros((self.n_state, n_obs_dense), dtype=x_pred_t.dtype)
                S_default = I_obs_dense * 1e6
                innovations_default = jnp.zeros((n_obs_dense,), dtype=ys_arr.dtype)
                innovation_cov_default = I_obs_dense * 1e6
            else:
                K_default = jnp.empty((self.n_state, 0), dtype=x_pred_t.dtype)
                S_default = jnp.empty((0, 0), dtype=ys_arr.dtype)
                innovations_default = jnp.empty((0,), dtype=ys_arr.dtype)
                innovation_cov_default = jnp.empty((0, 0), dtype=ys_arr.dtype)

            K_val = K_default
            S_obs_reg = S_default
            ll_t = jnp.array(-1e6, dtype=I_s.dtype) # Default LL contribution
            current_solve_ok = jnp.array(False) # Indicates if K gain solve was successful

            # Only perform update calculations if observation dimension > 0
            def do_dense_update_calculations(operand):
                 P_pred, C, H, I_obs, v_obs_calc = operand
                 PCt_obs = P_pred @ C.T
                 S_obs = C @ PCt_obs + H
                 S_obs_reg_calc = S_obs + kf_jitter * I_obs # Regularized innovation covariance

                 K_val_calc = jnp.zeros((self.n_state, n_obs_dense), dtype=P_pred.dtype)
                 current_solve_ok_calc = jnp.array(False)

                 # Attempt 1: Cholesky solve for K gain (S @ K.T = PCt)
                 try:
                     L_S_obs=jnp.linalg.cholesky(S_obs_reg_calc)
                     K_T_temp=jax.scipy.linalg.solve_triangular(L_S_obs,PCt_obs.T,lower=True,trans='N')
                     K_chol_attempt=jax.scipy.linalg.solve_triangular(L_S_obs.T,K_T_temp,lower=False,trans='N').T # Corrected solve direction
                     current_solve_ok_calc=jnp.all(jnp.isfinite(K_chol_attempt))
                     K_val_calc = jnp.where(current_solve_ok_calc, K_chol_attempt, K_val_calc)
                 except Exception:
                     current_solve_ok_calc=jnp.array(False)

                 # If Cholesky solve failed, attempt standard solve
                 def _fb_sol(combined_op):
                     op_data, _ = combined_op
                     PCt_op, S_reg_op = op_data
                     try: K_fb=jax.scipy.linalg.solve(S_reg_op,PCt_op.T,assume_a='pos').T;return K_fb,jnp.all(jnp.isfinite(K_fb))
                     except Exception:return jnp.zeros_like(PCt_op.T).T,jnp.array(False)
                 def _keep_K(combined_op):
                     _, K_status = combined_op
                     return K_status[0], K_status[1]
                 K_val_calc,current_solve_ok_calc=jax.lax.cond(current_solve_ok_calc,_keep_K,_fb_sol,operand=((PCt_obs,S_obs_reg_calc),(K_val_calc,current_solve_ok_calc)))

                 # If standard solve also failed, attempt pinv
                 def _fb_pinv(combined_op):
                     op_data, _ = combined_op
                     PCt_op,S_reg_op=op_data
                     try: K_pinv=PCt_op@jnp.linalg.pinv(S_reg_op,rcond=1e-6);return K_pinv,jnp.all(jnp.isfinite(K_pinv))
                     except Exception:return jnp.zeros_like(PCt_op.T).T,jnp.array(False)
                 K_val_calc,current_solve_ok_calc=jax.lax.cond(current_solve_ok_calc,_keep_K,_fb_pinv,operand=((PCt_obs,S_obs_reg_calc),(K_val_calc,current_solve_ok_calc)))

                 # Clip Kalman gain
                 K_val_calc=jnp.clip(K_val_calc,-1e3,1e3)

                 # Calculate log-likelihood contribution if K solve was successful
                 ll_t_calc = jnp.array(-1e6, dtype=P_pred.dtype)
                 def compute_ll_if_solve_ok_dense(operand_ll):
                     S_reg_ll, v_ll, n_obs_ll = operand_ll
                     try:
                        sign,log_det_S=jnp.linalg.slogdet(S_reg_ll)
                        valid_det = (sign > 0) & jnp.isfinite(log_det_S)

                        L_S_ll=jnp.linalg.cholesky(S_reg_ll)
                        z=jax.scipy.linalg.solve_triangular(L_S_ll,v_ll,lower=True)
                        mah_d=jnp.sum(z**2)
                        valid_mah_d = jnp.isfinite(mah_d)

                        log_pi_term = jnp.log(2*jnp.pi)*n_obs_ll
                        ll_term=-0.5*(log_pi_term+log_det_S+mah_d)

                        return jnp.where(valid_det & valid_mah_d, ll_term, jnp.array(-1e6, dtype=P_pred.dtype))
                     except Exception:
                        return jnp.array(-1e6, dtype=P_pred.dtype)

                 ll_t_calc = jax.lax.cond(
                    current_solve_ok_calc,
                    compute_ll_if_solve_ok_dense,
                    lambda op: jnp.array(-1e6, dtype=P_pred.dtype),
                    operand=(S_obs_reg_calc, v_obs_calc, n_obs_dense)
                 )

                 return K_val_calc, S_obs_reg_calc, ll_t_calc, current_solve_ok_calc
            
            def no_update_calculations(operand):
                # Return default values with correct shapes when no observations
                return (
                    K_default,  # Shape: (n_state, n_obs_dense) or (n_state, 0)
                    S_default,  # Shape: (n_obs_dense, n_obs_dense) or (0, 0)
                    ll_t,       # Scalar
                    current_solve_ok  # Scalar boolean
                )

            # Execute update calculations conditionally based on n_obs_dense > 0
            K_val, S_obs_reg, ll_t, current_solve_ok = jax.lax.cond(
                jnp.array(n_obs_dense > 0),
                do_dense_update_calculations,
                no_update_calculations,
                operand=(P_pred_t, C_dense, H_dense, I_obs_dense, v_obs)
            )

            # Apply update using the calculated K, but only if solve was successful AND n_obs_dense > 0
            x_filt_t,P_filt_t=x_pred_t,P_pred_t # Initialize with predicted values

            def apply_update_dense(op):
                K_,v_,x_,P_,C_,H_,I_s_=op
                x_up=K_@v_
                x_f=x_+x_up
                x_f=jnp.clip(x_f,-MAX_STATE_VALUE,MAX_STATE_VALUE)

                IKC=I_s_-K_@C_
                P_f=IKC@P_@IKC.T+K_@H_@K_.T

                P_f=(P_f+P_f.T)/2.0
                P_f=P_f+kf_jitter*I_s_
                return x_f,P_f

            def skip_update_dense(op):
                _,_,x_,P_,_,_,_=op
                return x_,P_

            # Conditionally apply update based on whether solve was successful and n_obs_dense > 0
            upd_cond = jnp.array(n_obs_dense > 0) & current_solve_ok
            x_filt_t,P_filt_t=jax.lax.cond(upd_cond,apply_update_dense,skip_update_dense,operand=(K_val,v_obs,x_pred_t,P_pred_t,C_dense,H_dense,I_s))

            # --- Prepare outputs for this time step ---
            # Handle potential NaNs/Infs resulting from failed operations
            x_pred_t_safe = jnp.where(jnp.isfinite(x_pred_t), x_pred_t, jnp.zeros_like(x_pred_t))
            P_pred_t_safe = jnp.where(jnp.all(jnp.isfinite(P_pred_t)), P_pred_t, I_s*1e6)
            x_filt_t_safe = jnp.where(jnp.isfinite(x_filt_t), x_filt_t, jnp.zeros_like(x_filt_t))
            P_filt_t_safe = jnp.where(jnp.all(jnp.isfinite(P_filt_t)), P_filt_t, I_s*1e6)

            P_pred_t_safe = (P_pred_t_safe + P_pred_t_safe.T) / 2.0
            P_filt_t_safe = (P_filt_t_safe + P_filt_t_safe.T) / 2.0

            # Innovations and Innovation Covariance - ensure consistent shapes
            innovations_out = jnp.where(jnp.array(n_obs_dense > 0) & current_solve_ok, v_obs, innovations_default)
            innovations_out = jnp.where(jnp.isfinite(innovations_out), innovations_out, innovations_default)

            innovation_cov_out = jnp.where(jnp.array(n_obs_dense > 0) & current_solve_ok, S_obs_reg, innovation_cov_default)
            innovation_cov_out = jnp.where(jnp.all(jnp.isfinite(innovation_cov_out)), innovation_cov_out, innovation_cov_default)
            if n_obs_dense > 0:
                innovation_cov_out = (innovation_cov_out + innovation_cov_out.T)/2.0 # Ensure symmetry

            ll_t_safe = jnp.where(jnp.isfinite(ll_t), ll_t, jnp.array(-1e6, dtype=I_s.dtype))

            out = {
                'x_pred': x_pred_t_safe,
                'P_pred': P_pred_t_safe,
                'x_filt': x_filt_t_safe,
                'P_filt': P_filt_t_safe,
                'innovations': innovations_out,
                'innovation_cov': innovation_cov_out,
                'log_likelihood_contributions': ll_t_safe
            }

            # The next carry is the filtered state and covariance
            return (x_filt_t_safe, P_filt_t_safe), out

        # Run the scan operation over time steps
        init_carry=(self.init_x,self.init_P)
        (_, _),scan_outputs=lax.scan(step_dense_data,init_carry,ys_arr)

        # Final safety checks for NaNs/Infs and symmetry on stacked arrays
        # This is the same logic as in Kalman_filter_jax.filter's end section
        # ... [copy-paste final check logic from Kalman_filter_jax.filter here] ...
        # Ensure P_pred and P_filt are symmetric and finite in the final stacked arrays
        if 'P_pred' in scan_outputs:
             P_pred_final = scan_outputs['P_pred']
             P_pred_final = (P_pred_final + jnp.transpose(P_pred_final, (0, 2, 1))) / 2.0
             scan_outputs['P_pred'] = jnp.where(jnp.all(jnp.isfinite(P_pred_final), axis=(-2,-1), keepdims=True), P_pred_final, self.I_s[None,:,:]*1e6)

        if 'P_filt' in scan_outputs:
             P_filt_final = scan_outputs['P_filt']
             P_filt_final = (P_filt_final + jnp.transpose(P_filt_final, (0, 2, 1))) / 2.0
             scan_outputs['P_filt'] = jnp.where(jnp.all(jnp.isfinite(P_filt_final), axis=(-2,-1), keepdims=True), P_filt_final, self.I_s[None,:,:]*1e6)

        if 'innovation_cov' in scan_outputs and n_obs_dense > 0:
             S_final = scan_outputs['innovation_cov']
             S_final = (S_final + jnp.transpose(S_final, (0, 2, 1))) / 2.0
             I_obs_dense_3d = self.I_obs_full[None, :, :] # Add time dimension
             scan_outputs['innovation_cov'] = jnp.where(jnp.all(jnp.isfinite(S_final), axis=(-2,-1), keepdims=True), S_final, I_obs_dense_3d*1e6)
        elif 'innovation_cov' in scan_outputs and n_obs_dense == 0:
             # Ensure innovation_cov is empty if n_obs_dense is 0
             scan_outputs['innovation_cov'] = jnp.empty((T_steps, 0, 0), dtype=ys_arr.dtype)


        # Ensure LL contributions are finite
        if 'log_likelihood_contributions' in scan_outputs:
             scan_outputs['log_likelihood_contributions'] = jnp.where(
                 jnp.isfinite(scan_outputs['log_likelihood_contributions']),
                 scan_outputs['log_likelihood_contributions'],
                 jnp.full_like(scan_outputs['log_likelihood_contributions'], -1e6)
             )

        # Ensure state outputs are finite
        if 'x_pred' in scan_outputs:
             scan_outputs['x_pred'] = jnp.where(jnp.isfinite(scan_outputs['x_pred']), scan_outputs['x_pred'], jnp.zeros_like(scan_outputs['x_pred']))
        if 'x_filt' in scan_outputs:
             scan_outputs['x_filt'] = jnp.where(jnp.isfinite(scan_outputs['x_filt']), scan_outputs['x_filt'], jnp.zeros_like(scan_outputs['x_filt']))
        if 'innovations' in scan_outputs:
             scan_outputs['innovations'] = jnp.where(jnp.isfinite(scan_outputs['innovations']), scan_outputs['innovations'], jnp.zeros_like(scan_outputs['innovations']))


        return scan_outputs

    # --- RTS Smoother for original data (or simulated data) ---
    # This smoother implementation is also for DENSE filter results.
    def _rts_smoother_backend(self, filter_results_dict: Dict) -> Tuple[jax.Array, jax.Array]:
        """
        Internal RTS smoother implementation for filter results from dense data.
        Handles potential NaNs/Infs in dense filter outputs.
        This should be identical to the one in Kalman_filter_jax for dense data.
        """
        x_pred = filter_results_dict['x_pred'] # T x n_state
        P_pred = filter_results_dict['P_pred'] # T x n_state x n_state
        x_filt = filter_results_dict['x_filt'] # T x n_state
        P_filt = filter_results_dict['P_filt'] # T x n_state x n_state

        T_mat = self.T # n_state x n_state
        N = x_filt.shape[0] # Number of time steps

        # Handle empty time series case
        if N == 0: return jnp.empty((0, self.n_state), dtype=x_filt.dtype), jnp.empty((0, self.n_state, self.n_state), dtype=P_filt.dtype)

        kf_jitter_smooth_backend = _KF_JITTER_DK
        I_s = self.I_s # Identity matrix for state space

        # Handle potential infinities/NaNs in filter outputs before smoothing
        # Use zeros/identity with large scale as safe defaults
        x_pred_safe = jnp.where(jnp.isfinite(x_pred), x_pred, jnp.zeros_like(x_pred))
        P_pred_safe = jnp.where(jnp.all(jnp.isfinite(P_pred), axis=(-2,-1), keepdims=True), P_pred, I_s[None,:,:] * 1e6)
        x_filt_safe = jnp.where(jnp.isfinite(x_filt), x_filt, jnp.zeros_like(x_filt))
        P_filt_safe = jnp.where(jnp.all(jnp.isfinite(P_filt), axis=(-2,-1), keepdims=True), P_filt, I_s[None,:,:] * 1e6)

        # Ensure covariances are symmetric before smoothing
        P_pred_safe = (P_pred_safe + jnp.transpose(P_pred_safe, (0, 2, 1))) / 2.0
        P_filt_safe = (P_filt_safe + jnp.transpose(P_filt_safe, (0, 2, 1))) / 2.0

        # Smoothed state/cov at the last time step (N-1) are the filtered ones.
        x_s_N_minus_1 = x_filt_safe[N - 1]
        P_s_N_minus_1 = P_filt_safe[N - 1]

        # Prepare scan inputs for backward pass
        P_pred_for_scan = P_pred_safe[1:N]
        P_filt_for_scan = P_filt_safe[0:N - 1]
        x_pred_for_scan = x_pred_safe[1:N]
        x_filt_for_scan = x_filt_safe[0:N - 1]

        # Reverse the inputs for lax.scan
        scan_inputs = (
            P_pred_for_scan[::-1], # P_pred[N-1], ..., P_pred[1]
            P_filt_for_scan[::-1], # P_filt[N-2], ..., P_filt[0]
            x_pred_for_scan[::-1], # x_pred[N-1], ..., x_pred[1]
            x_filt_for_scan[::-1]  # x_filt[N-2], ..., x_filt[0]
        )

        def backward_step_common(carry_smooth, scan_t_rev_idx):
            # carry_smooth: (x_s_next_t, P_s_next_t) = smoothed state/cov at time t+1
            # scan_t_rev_idx: (Pp_next_t, Pf_t, xp_next_t, xf_t) = data for time t and t+1
            x_s_next_t, P_s_next_t = carry_smooth
            Pp_next_t, Pf_t, xp_next_t, xf_t = scan_t_rev_idx

            # --- Compute Smoother Gain J_t ---
            # J_t = P_filt_t @ T.T @ P_pred_{t+1}^-1
            Pf_t_sym = (Pf_t + Pf_t.T) / 2.0
            Pp_next_reg_val = Pp_next_t + kf_jitter_smooth_backend * jnp.eye(self.n_state, dtype=Pp_next_t.dtype)

            Jt_res = jnp.zeros((self.n_state, self.n_state), dtype=Pf_t.dtype)
            current_J_solve_ok = jnp.array(False)

            # Attempt 1: Solve using jax.scipy.linalg.solve
            try:
                Jt_T = jax.scipy.linalg.solve(Pp_next_reg_val, (T_mat @ Pf_t_sym).T, assume_a='pos')
                Jt_res = Jt_T.T
                current_J_solve_ok = jnp.all(jnp.isfinite(Jt_res))
            except Exception:
                current_J_solve_ok = jnp.array(False)

            # If standard solve failed, attempt pinv
            def _fb_pinv_J_common(operand_tuple_J_pinv):
                matrices_for_pinv, _ = operand_tuple_J_pinv
                T_loc, Pf_s_loc, Pp_n_r_loc = matrices_for_pinv
                try:
                    J_pinv = (Pf_s_loc @ T_loc.T) @ jnp.linalg.pinv(Pp_n_r_loc, rcond=1e-6)
                    return J_pinv, jnp.all(jnp.isfinite(J_pinv))
                except Exception:
                    return jnp.zeros_like(Pf_s_loc), jnp.array(False)

            def _keep_J_common(operand_tuple_J_keep):
                _, J_and_status_to_keep = operand_tuple_J_keep
                return J_and_status_to_keep[0], J_and_status_to_keep[1]

            Jt_res, current_J_solve_ok = jax.lax.cond(
                current_J_solve_ok,
                _keep_J_common,
                _fb_pinv_J_common,
                operand=((T_mat, Pf_t_sym, Pp_next_reg_val), (Jt_res, current_J_solve_ok))
            )

            # Ensure Jt_res is finite; if not, set to zero
            Jt_res_safe = jnp.where(jnp.isfinite(Jt_res), Jt_res, jnp.zeros_like(Jt_res))

            # --- Compute Smoothed State and Covariance ---
            # x_s_t = x_filt_t + J_t @ (x_s_{t+1} - x_pred_{t+1})
            # P_s_t = P_filt_t + J_t @ (P_s_{t+1} - P_pred_{t+1}) @ J_t.T

            x_d = x_s_next_t - xp_next_t
            P_d = P_s_next_t - Pp_next_t

            x_s_t = xf_t + Jt_res_safe @ x_d
            P_s_t = Pf_t_sym + Jt_res_safe @ P_d @ Jt_res_safe.T

            # Ensure P_s_t is symmetric and handle NaNs/Infs
            P_s_t = (P_s_t + P_s_t.T) / 2.0
            P_s_t = jnp.where(jnp.all(jnp.isfinite(P_s_t), axis=(-2,-1), keepdims=True), P_s_t, I_s * 1e6)

            # Handle NaNs/Infs in state mean
            x_s_t = jnp.where(jnp.isfinite(x_s_t), x_s_t, jnp.zeros_like(x_s_t))

            # The carry for the next step (t-1) is the smoothed state/cov at the current step (t)
            return (x_s_t, P_s_t), (x_s_t, P_s_t)

        # Initial carry for the backward scan: smoothed state/cov at time N-1
        init_carry_smooth = (x_s_N_minus_1, P_s_N_minus_1)

        # Run the backward scan
        # The outputs (x_s_rev, P_s_rev) will be the smoothed states/covariances
        # from time N-2 down to 0, stacked in reverse order.
        (_, _), (x_s_rev, P_s_rev) = lax.scan(backward_step_common, init_carry_smooth, scan_inputs)

        # Concatenate the smoothed states/covariances:
        # Reverse the scan outputs (time N-2 to 0) and prepend time N-1
        x_smooth = jnp.concatenate([x_s_rev[::-1], x_filt_safe[N - 1:N]], axis=0)
        P_smooth = jnp.concatenate([P_s_rev[::-1], P_filt_safe[N - 1:N]], axis=0)

        # Final safety check for NaNs/Infs and symmetry after concatenation
        x_smooth = jnp.where(jnp.isfinite(x_smooth), x_smooth, jnp.zeros_like(x_smooth))
        P_smooth = jnp.where(jnp.all(jnp.isfinite(P_smooth), axis=(-2,-1), keepdims=True), P_smooth, I_s[None,:,:] * 1e6)
        P_smooth = (P_smooth + jnp.transpose(P_smooth, (0, 2, 1))) / 2.0

        return x_smooth, P_smooth


    # --- Simulation Draw Methods ---
    # These methods are adapted for the Durbin-Koopman approach
    # and rely on the _filter_internal and _rts_smoother_backend above,
    # which are assumed to work correctly for DENSE data.

# Fix 1: Correct the Durbin-Koopman formula in hybrid_dk_smoother.py

    def _draw_simulation_small(self, original_ys_dense: jax.Array, key: jax.random.PRNGKey,
                            x_smooth_original_dense: jax.Array) -> jax.Array:
        """
        Single simulation draw for small state dimensions using the D-K approach.
        FIXED: Corrected Durbin-Koopman formula
        """
        T_steps = original_ys_dense.shape[0]
        if T_steps == 0: return jnp.empty((0, self.n_state), dtype=self.init_x.dtype)

        n_s, n_eps_shocks = self.n_state, self.n_shocks

        # Split keys for different random components
        key_init, key_eps, key_eta = random.split(key, 3)

        # Generate initial state draw from N(init_x, init_P)
        if self.init_chol_success:
            z0 = random.normal(key_init, (n_s,), dtype=self.init_x.dtype)
            x0_star = self.init_x + self.L0 @ z0
        else:
            x0_star = self.init_x

        # Generate state shocks
        if n_eps_shocks > 0:
            eps_star_sim = random.normal(key_eps, (T_steps, n_eps_shocks), dtype=self.R.dtype)
        else:
            eps_star_sim = jnp.zeros((T_steps, 0), dtype=self.R.dtype)

        # Simulate state path with shocks
        def state_step_with_shocks(x_prev, eps_t):
            shock = self.R @ eps_t if n_eps_shocks > 0 else jnp.zeros(self.n_state, dtype=x_prev.dtype)
            x_t = self.T @ x_prev + shock
            x_t = jnp.clip(x_t, -1e6, 1e6)
            return x_t, x_t

        _, x_star_path = lax.scan(state_step_with_shocks, x0_star, eps_star_sim)

        # Generate simulated observations
        y_star_dense_sim = jnp.zeros((T_steps, self.n_obs_full), dtype=x_star_path.dtype)
        if self.n_obs_full > 0:
            eta_star_full_sim = self._simulate_obs_noise(key_eta, (T_steps,))
            y_star_dense_sim = (x_star_path @ self.C_full.T) + eta_star_full_sim

        # Run filter and smoother on simulated data
        filter_results_star = self._filter_internal(y_star_dense_sim)
        x_smooth_star_dense, _ = self._rts_smoother_backend(filter_results_star)

        # FIXED: Correct Durbin-Koopman formula
        # OLD (INCORRECT): x_draw = x_star_path + (x_smooth_original_dense - x_smooth_star_dense)
        # NEW (CORRECT):   x_draw = x_smooth_original_dense + (x_star_path - x_smooth_star_dense)
        x_draw = x_smooth_original_dense + (x_star_path - x_smooth_star_dense)

        # Ensure final draw is finite
        x_draw = jnp.where(jnp.isfinite(x_draw), x_draw, jnp.zeros_like(x_draw))

        return x_draw


    # Note: The code for _draw_simulation_large is identical to _draw_simulation_small.
    # This suggests the performance difference is handled by the vmap vs loop strategy
    # in run_smoother_draws, not fundamentally different simulation/smoothing steps here.
    def _draw_simulation_large(self, original_ys_dense: jax.Array, key: jax.random.PRNGKey,
                              x_smooth_original_dense: jax.Array) -> jax.Array:
         """
         Single simulation draw for potentially large state dimensions.
         Currently identical implementation to _draw_simulation_small.
         Assumes original_ys_dense contains DENSE observations (no NaNs).
         """
         Tsteps = original_ys_dense.shape[0]
         if Tsteps == 0: return jnp.empty((0, self.n_state), dtype=self.init_x.dtype)

         n_s, n_eps_shocks = self.n_state, self.n_shocks
         kf_jitter = _KF_JITTER_DK # Using jitter from DK class

         key_init, key_eps, key_eta = random.split(key, 3)

         # Generate initial state draw
         try:
             init_P_reg = self.init_P + kf_jitter * self.I_s
             L0 = jnp.linalg.cholesky(init_P_reg); z0 = random.normal(key_init, (n_s,), dtype=self.init_x.dtype)
             x0_star = self.init_x + L0 @ z0
         except Exception:
             print("Warning: Initial P Cholesky failed in _draw_simulation_large. Using init_x mean.")
             x0_star = self.init_x

         # Generate state shocks
         eps_star_sim = random.normal(key_eps,(Tsteps,n_eps_shocks),dtype=self.R.dtype) if n_eps_shocks>0 else jnp.zeros((Tsteps,0),dtype=self.R.dtype)

         # Simulate state path
         def state_sim_step(x_prev_star, eps_t_star_arg):
             shock_term=self.R@eps_t_star_arg if n_eps_shocks>0 else jnp.zeros(self.n_state,dtype=x_prev_star.dtype)
             x_curr_star=self.T@x_prev_star+shock_term
             x_curr_star=jnp.clip(x_curr_star,-1e6,1e6)
             return x_curr_star,x_curr_star
         _,x_star_path=lax.scan(state_sim_step,x0_star,eps_star_sim)

         # Generate simulated observations
         y_star_dense_sim=jnp.zeros((Tsteps,self.n_obs_full),dtype=x_star_path.dtype)
         if self.n_obs_full>0:
             eta_star_full_sim=self._simulate_obs_noise(key_eta,(Tsteps,)) # Use internal noise sim method
             y_star_dense_sim=(x_star_path@self.C_full.T)+eta_star_full_sim # Use full C_full

         # Run filter and smoother on simulated data
         filter_results_star_dense = self._filter_internal(y_star_dense_sim)
         x_smooth_star_dense, _ = self._rts_smoother_backend(filter_results_star_dense)

         # Compute final draw
         #x_draw = x_star_path + (x_smooth_original_dense - x_smooth_star_dense)
         x_draw = x_smooth_original_dense + (x_star_path - x_smooth_star_dense)

         # Ensure final draw is finite
         x_draw = jnp.where(jnp.isfinite(x_draw), x_draw, jnp.zeros_like(x_draw))
         
         return x_draw


    # --- Public Methods ---
    def filter(self, ys_dense: jax.Array) -> Dict[str, jax.Array]:
        """
        Public method to run Kalman filter on DENSE observations.
        Note: This filter assumes `ys_dense` has no NaNs or missing values.
        If your original data has NaNs, you need to handle them before calling this,
        or use a more sophisticated filter that supports time-varying missing data.
        """
        return self._filter_internal(ys_dense)

    def smooth(self, filter_results: Dict) -> Tuple[jax.Array, jax.Array]:
        """
        Public method to run RTS smoother on filter results (presumably from dense data).
        """
        return self._rts_smoother_backend(filter_results)

    def draw_simulation(self, original_ys_dense: jax.Array, key: jax.random.PRNGKey,
                       x_smooth_original_dense: jax.Array) -> jax.Array:
        """
        Draw a single simulation based on original dense data and smoothed states.
        Selects internal implementation based on state dimension (currently same).
        Assumes original_ys_dense is DENSE.
        """
        # The internal implementations are currently the same, the flag only affects vmap strategy
        # return self._draw_simulation_small(original_ys_dense, key, x_smooth_original_dense)
        # Let's just call one directly as the code is identical, or keep the flag for potential future divergence.
        if self.use_small_implementation:
            return self._draw_simulation_small(original_ys_dense, key, x_smooth_original_dense)
        else:
             # In the original code, 'large' might use a different algorithm.
             # Here it's the same code, only the *batching* strategy in run_smoother_draws changes.
             # Keep the distinction just in case, although it calls the same logic.
            return self._draw_simulation_large(original_ys_dense, key, x_smooth_original_dense)


    def run_smoother_draws(self,
                          original_ys_dense: jax.Array,
                          key: jax.random.PRNGKey,
                          num_draws: int,
                          x_smooth_rts_original_dense: jax.Array) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        """
        Run multiple simulation draws efficiently.
        Selects batching strategy based on state dimension.
        Assumes original_ys_dense is DENSE and x_smooth_rts_original_dense is from DENSE data.

        Args:
            original_ys_dense: Original dense observations. Shape (T, n_obs_full).
            key: JAX random key.
            num_draws: Number of simulation draws.
            x_smooth_rts_original_dense: Smoothed states from original dense data. Shape (T, n_state).

        Returns:
            If num_draws == 1: Single draw (T, n_state).
            If num_draws > 1: (mean_draws, median_draws, all_draws).
                             Shapes: (T, n_state), (T, n_state), (num_draws, T, n_state).
        """
        #print(f"Running {num_draws} simulation draws using {'small' if self.use_small_implementation else 'large'} state strategy...")
        # start_time = time.time() # Uncomment for profiling

        # Input validation
        if num_draws <= 0:
            raise ValueError("Number of draws must be positive")

        T_steps = original_ys_dense.shape[0]

        # Handle empty case
        if T_steps == 0:
            empty_shape_mean = (T_steps, self.n_state)
            empty_shape_all = (num_draws, T_steps, self.n_state)

            if num_draws == 1:
                return jnp.empty(empty_shape_mean, dtype=_DEFAULT_DTYPE)
            else:
                return (
                    jnp.empty(empty_shape_mean, dtype=_DEFAULT_DTYPE),
                    jnp.empty(empty_shape_mean, dtype=_DEFAULT_DTYPE),
                    jnp.empty(empty_shape_all, dtype=_DEFAULT_DTYPE)
                )

        # Generate keys for each draw
        keys = random.split(key, num_draws)

        # Choose batching strategy
        # For small state dims, vmap is usually efficient.
        # For very large state dims, sequential processing might manage memory better,
        # or avoid large compiled jaxprs from vmap over the draw dimension.
        if self.use_small_implementation:
            # For small state dimensions, use regular vmap over the `keys` axis (0)
            vmapped_draw = vmap(
                lambda k: self.draw_simulation(original_ys_dense, k, x_smooth_rts_original_dense),
                in_axes=(0,) # vmap over the first axis of keys
            )
            all_draws = vmapped_draw(keys)
        else:
            # For large state dimensions, loop sequentially (avoids large vmap)
            # This loop cannot be jit-compiled with the draw function inside it
            # if num_draws is not static.
            # If num_draws is static, vmap might still be an option.
            # If num_draws is NOT static, a Python loop calling the JITted draw function is needed.
            # Let's assume num_draws can be non-static and use a Python loop.
            # Note: This means run_smoother_draws itself cannot be jit-compiled effectively
            # if this branch is taken and num_draws is not static.

            # If num_draws is small and static, vmap might be fine even for "large" state.
            # If num_draws is large and non-static, this loop is the way.
            # Let's prioritize the non-static case and use a python loop.

            all_draws = jnp.zeros((num_draws, T_steps, self.n_state), dtype=_DEFAULT_DTYPE)

            # JIT compile the single draw function once outside the loop
            jit_draw_simulation = jax.jit(self.draw_simulation)

            for i in range(num_draws):
                # Call the JITted single draw function for each key
                draw = jit_draw_simulation(original_ys_dense, keys[i], x_smooth_rts_original_dense)
                all_draws = all_draws.at[i].set(draw)
                # Optional: print progress
                # if (i+1) % 10 == 0 or i+1 == num_draws:
                #     print(f"Completed {i+1}/{num_draws} draws...")

        # end_time = time.time() # Uncomment for profiling
        # print(f"Completed {num_draws} draws in {end_time - start_time:.2f} seconds")


        # Return results
        if num_draws == 1:
            return all_draws[0] # Return shape (T, n_state)
        else:
            # Compute statistics over the draw dimension (axis=0)
            mean_draws = jnp.mean(all_draws, axis=0)

            # Median can be computationally expensive for many draws/large state.
            # For very large state dimensions, mean might be a practical substitute for median.
            # The threshold here might need adjustment.
            if self.n_state >= 200: # Example threshold for using mean as median proxy
                median_draws = mean_draws
            else:
                # Compute median along the draw axis
                median_draws = jnp.median(all_draws, axis=0)

            return mean_draws, median_draws, all_draws

    # Public high-level function to run smoothing and then draws
    def smooth_and_simulate(self,
                          original_ys_dense: jax.Array,
                          key: jax.random.PRNGKey,
                          num_draws: int) -> Tuple[jax.Array, Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]]:
        """
        One-step method to smooth original dense data and run simulation draws.

        Args:
            original_ys_dense: Original dense observations. Shape (T, n_obs_full).
                               Assumed to have no NaNs for filtering/smoothing.
            key: JAX random key for simulation draws.
            num_draws: Number of simulation draws.

        Returns:
            Tuple: (smoothed_states, simulation_results)
            smoothed_states: Smoothed state means (T, n_state).
            simulation_results: Depends on num_draws:
                               If num_draws == 1: Single draw (T, n_state).
                               If num_draws > 1: Tuple (mean_draws, median_draws, all_draws)
                                                Shapes: (T, n_state), (T, n_state), (num_draws, T, n_state).
        """
        start_time = time.time()
        # print("Running filter and smoother on original dense data...")

        # Run filter on original dense data
        filter_results = self.filter(original_ys_dense)
        # Run smoother on filter results
        smoothed_states, _ = self.smooth(filter_results) # We only need the smoothed mean for draws

        filter_smooth_time = time.time() - start_time
        # print(f"Filter and smoother completed in {filter_smooth_time:.2f} seconds")

        # Generate simulation draws
        # print(f"Generating {num_draws} simulation draws...")
        sim_results = self.run_smoother_draws(
            original_ys_dense, key, num_draws, smoothed_states
        )

        total_time = time.time() - start_time
        # print(f"Total smooth and simulate process completed in {total_time:.2f} seconds")

        return smoothed_states, sim_results


# --- END OF FILE utils/hybrid_dk_smoother.py ---