import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax, random, vmap
from jax.typing import ArrayLike
import numpy as onp
from typing import Tuple, Optional, Union, Sequence, Dict, Any

# Configure JAX for float64 if not already
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


_MACHINE_EPSILON = jnp.finfo(_DEFAULT_DTYPE).eps
_KF_JITTER = 1e-8

class KalmanFilter:
    def __init__(self, T: ArrayLike, R: ArrayLike, C: ArrayLike, H: ArrayLike, init_x: ArrayLike, init_P: ArrayLike):
        desired_dtype = _DEFAULT_DTYPE
        self.T = jnp.asarray(T, dtype=desired_dtype)
        self.R = jnp.asarray(R, dtype=desired_dtype)
        self.C = jnp.asarray(C, dtype=desired_dtype)
        self.H = jnp.asarray(H, dtype=desired_dtype) # This is the observation noise covariance matrix
        self.init_x = jnp.asarray(init_x, dtype=desired_dtype)
        self.init_P = jnp.asarray(init_P, dtype=desired_dtype)

        n_state = self.T.shape[0]
        n_obs_full = self.C.shape[0] # C maps state to the *full* observation space
        n_shocks = self.R.shape[1] if self.R.ndim == 2 and self.R.shape[1] > 0 else 0

        if self.T.shape != (n_state, n_state): raise ValueError(f"T shape mismatch: {self.T.shape} vs ({n_state}, {n_state})")
        if n_shocks > 0 and self.R.shape != (n_state, n_shocks): raise ValueError(f"R shape mismatch: {self.R.shape} vs ({n_state}, {n_shocks})")
        elif n_shocks == 0 and self.R.size != 0: self.R = jnp.zeros((n_state, 0), dtype=desired_dtype)
        if self.C.shape != (n_obs_full, n_state): raise ValueError(f"C shape mismatch: {self.C.shape} vs ({n_obs_full}, {n_state})")
        # H should be n_obs_full x n_obs_full, as it's the cov matrix for *full* observation noise
        if self.H.shape != (n_obs_full, n_obs_full): raise ValueError(f"H shape mismatch: {self.H.shape} vs ({n_obs_full}, {n_obs_full})")
        if self.init_x.shape != (n_state,): raise ValueError(f"init_x shape mismatch: {self.init_x.shape} vs ({n_state},)")
        if self.init_P.shape != (n_state, n_state): raise ValueError(f"init_P shape mismatch: {self.init_P.shape} vs ({n_state}, {n_state})")

        self.n_state = n_state
        self.n_obs_full = n_obs_full # Dimension of the *potential* observation vector
        self.n_shocks = n_shocks
        self.I_s = jnp.eye(self.n_state, dtype=desired_dtype)
        self.I_obs_full = jnp.eye(self.n_obs_full, dtype=desired_dtype) if self.n_obs_full > 0 else jnp.empty((0, 0), dtype=desired_dtype)

        if self.n_shocks > 0:
            # state_cov = R @ R.T is the covariance of the state *shocks* (eta_t in x_t = T x_{t-1} + R eta_t)
            self.state_cov = self.R @ self.R.T
        else:
            self.state_cov = jnp.zeros((self.n_state, self.n_state), dtype=desired_dtype)

        # Pre-compute for simulate_obs_noise (if needed by other methods, though filter doesn't use it)
        # H_full needs to be PSD. Add jitter.
        if self.n_obs_full > 0:
            H_reg_chol = self.H + _KF_JITTER * self.I_obs_full
            try:
                self.L_H_full = jnp.linalg.cholesky(H_reg_chol)
                self.simulate_obs_noise_internal = self._simulate_obs_noise_chol_internal
            except Exception:
                self.H_stable_full = self.H + _KF_JITTER * 10 * self.I_obs_full # Use larger jitter if Cholesky fails? Or just stable H?
                self.simulate_obs_noise_internal = self._simulate_obs_noise_mvn_internal
        else:
             # Define dummy simulate_obs_noise for n_obs_full == 0
             self.simulate_obs_noise_internal = lambda key, shape: jnp.empty(tuple(shape) + (0,), dtype=desired_dtype)


    # These simulate_obs_noise methods are only used by the simulate_state_space function,
    # which is outside the class. Let's keep them internal in case they are needed later,
    # but note they aren't used by the filter method itself.
    def _simulate_obs_noise_chol_internal(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        if self.n_obs_full == 0: return jnp.empty(tuple(shape) + (0,), dtype=self.H.dtype)
        z_eta = random.normal(key, tuple(shape) + (self.n_obs_full,), dtype=self.H.dtype)
        return z_eta @ self.L_H_full.T

    def _simulate_obs_noise_mvn_internal(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        if self.n_obs_full == 0: return jnp.empty(tuple(shape) + (0,), dtype=self.H.dtype)
        mvn_shape = tuple(shape) if len(shape) > 0 else ()
        try: return random.multivariate_normal(key, jnp.zeros((self.n_obs_full,), dtype=self.H.dtype),self.H_stable_full, shape=mvn_shape, dtype=self.H.dtype)
        except Exception: return jnp.zeros(tuple(shape) + (self.n_obs_full,), dtype=self.H.dtype)


    # The filter method receives *full* observation vectors `ys` (shape T x n_obs_full)
    # and relies on the static_* arguments to select the currently observed subset.
    # This implies the subset of observed variables is CONSTANT over time.
    # If observations are missing arbitrarily over time, this filter needs modification
    # (e.g., passing time-varying C_t, H_t, I_obs_t or a time-varying mask).
    # Assuming the provided filter is sufficient for the user's use case (constant observed subset).
    def filter(self, ys: ArrayLike, static_valid_obs_idx: jax.Array, static_n_obs_actual: int, static_C_obs: jax.Array, static_H_obs: jax.Array, static_I_obs: jax.Array) -> Dict[str, jax.Array]:
        """
        Runs the Kalman filter on the provided data.

        Args:
            ys: Observation data, shape (time_steps, n_obs_full). Can contain NaNs.
            static_valid_obs_idx: 1D array of indices in the full observation vector
                                  that are actually observed at each time step. Shape (n_obs_actual,).
            static_n_obs_actual: The number of actually observed variables (static_valid_obs_idx.shape[0]).
            static_C_obs: The observation matrix C restricted to the observed rows. Shape (n_obs_actual, n_state).
                          This is C[static_valid_obs_idx, :].
            static_H_obs: The observation noise covariance H restricted to the observed rows/cols. Shape (n_obs_actual, n_obs_actual).
                          This is H[static_valid_obs_idx[:, None], static_valid_obs_idx].
            static_I_obs: Identity matrix of size n_obs_actual. Shape (n_obs_actual, n_obs_actual).

        Returns:
            Dictionary of filter results (x_pred, P_pred, x_filt, P_filt, innovations,
            innovation_cov, log_likelihood_contributions).
        """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype) # Use dtype of C which is _DEFAULT_DTYPE
        T_mat, I_s = self.T, self.I_s
        state_cov = self.state_cov # This is Q, the state shock covariance (R @ R.T)
        kf_jitter = _KF_JITTER
        MAX_STATE_VALUE = 1e6
        T_steps = ys_arr.shape[0]

        # Handle empty time series case
        if T_steps == 0:
            # Need to return empty arrays with correct shapes and dtypes,
            # especially for the observation-dependent outputs.
            # If n_obs_actual is 0, dimensions should be 0.
            obs_dim = static_n_obs_actual if static_n_obs_actual > 0 else 0
            return {
                'x_pred': jnp.empty((0, self.n_state), dtype=I_s.dtype),
                'P_pred': jnp.empty((0, self.n_state, self.n_state), dtype=I_s.dtype),
                'x_filt': jnp.empty((0, self.n_state), dtype=I_s.dtype),
                'P_filt': jnp.empty((0, self.n_state, self.n_state), dtype=I_s.dtype),
                'innovations': jnp.empty((0, obs_dim), dtype=ys_arr.dtype), # Innovations have same dtype as ys
                'innovation_cov': jnp.empty((0, obs_dim, obs_dim), dtype=ys_arr.dtype), # Innovation cov has same dtype as ys
                'log_likelihood_contributions': jnp.empty((0,), dtype=I_s.dtype) # LL contributions have same dtype as state (float64)
            }

        def step_nan_handling(carry, y_t_full_slice):
            # carry: (x_prev_filt, P_prev_filt)
            # y_t_full_slice: observation vector at time t, shape (n_obs_full,)

            x_prev_filt, P_prev_filt = carry

            # --- Prediction Step ---
            x_pred_t = T_mat @ x_prev_filt
            x_pred_t = jnp.clip(x_pred_t, -MAX_STATE_VALUE, MAX_STATE_VALUE)

            # Ensure P_prev_filt is symmetric and regularized before prediction
            P_prev_filt_sym = (P_prev_filt + P_prev_filt.T) / 2.0
            P_prev_filt_reg = P_prev_filt_sym + kf_jitter * I_s

            P_pred_t = T_mat @ P_prev_filt_reg @ T_mat.T + state_cov
            # Ensure P_pred_t is symmetric and regularized after prediction
            P_pred_t = (P_pred_t + P_pred_t.T) / 2.0
            P_pred_t = P_pred_t + kf_jitter * I_s

            # --- Update Step ---
            # Select the actually observed part of y_t using static indices
            y_obs_t = jnp.take(y_t_full_slice, static_valid_obs_idx, axis=0) if static_n_obs_actual > 0 else jnp.empty((0,), dtype=y_t_full_slice.dtype)

            # Check if *any* value in y_obs_t is finite. If not, skip update.
            # This handles time steps where all relevant observations are NaN,
            # even if static_n_obs_actual > 0.
            perform_update = jnp.array(static_n_obs_actual > 0) & jnp.any(jnp.isfinite(y_obs_t))
            # Replace NaNs in y_obs_t with 0 for calculations where it's needed (e.g., innovation)
            # The innovation calculation below will handle NaNs correctly if `perform_update` is False.
            y_obs_t_clean = jnp.nan_to_num(y_obs_t, nan=0.0) # Replace NaN with 0 for calculations

            y_pred_obs = static_C_obs @ x_pred_t
            v_obs = y_obs_t_clean - y_pred_obs # Calculate innovation using cleaned obs

            # Initialize K, S, log-likelihood contribution
            K_res = jnp.zeros((self.n_state, static_n_obs_actual if static_n_obs_actual > 0 else 0), dtype=x_pred_t.dtype)
            S_obs_reg_val = static_I_obs * 1e6 if static_n_obs_actual > 0 else jnp.empty((0,0), dtype=ys_arr.dtype) # Default S value for non-update case
            ll_t = jnp.array(-1e6, dtype=I_s.dtype) # Default LL contribution for non-update case
            solve_status = jnp.array(False) # Indicates if K gain solve was successful

            # Only perform update calculations if there are observed finite values
            def do_update_calculations(operand):
                P_pred, C_obs, H_obs, I_obs, v_obs_calc = operand
                PCt_obs_val = P_pred @ C_obs.T
                S_obs_val = C_obs @ PCt_obs_val + H_obs
                S_obs_reg_val_calc = S_obs_val + kf_jitter * I_obs

                K_res_calc = jnp.zeros((self.n_state, static_n_obs_actual), dtype=P_pred.dtype)
                current_solve_ok_calc = jnp.array(False)

                # Attempt 1: Cholesky solve for Kalman Gain K
                try:
                    L_S_obs = jnp.linalg.cholesky(S_obs_reg_val_calc)
                    # Solves L_S_obs @ K_T_temp = PCt_obs_val.T
                    K_T_temp = jax.scipy.linalg.solve_triangular(L_S_obs, PCt_obs_val.T, lower=True, trans='N')
                    # Solves L_S_obs.T @ K_res_calc.T = K_T_temp => K_res_calc.T = (L_S_obs.T)^-1 @ K_T_temp
                    # K_res_calc = K_T_temp.T @ L_S_obs^-1
                    K_chol_attempt = jax.scipy.linalg.solve_triangular(L_S_obs.T, K_T_temp, lower=False, trans='N').T # Corrected solve direction
                    current_solve_ok_calc = jnp.all(jnp.isfinite(K_chol_attempt))
                    K_res_calc = jnp.where(current_solve_ok_calc, K_chol_attempt, K_res_calc) # Use chol result if ok
                except Exception:
                    current_solve_ok_calc = jnp.array(False) # Chol failed

                # If Cholesky solve failed, attempt standard solve (S @ K.T = PCt)
                def _standard_solve_branch_local(operand_tuple_std_solve):
                    # operand_tuple is ((matrices_for_solve), (K_previous, status_previous))
                    matrices_for_solve_std, _ = operand_tuple_std_solve
                    PCt_op_std, S_reg_op_std = matrices_for_solve_std
                    try:
                        # Solves S_reg_op_std @ K_std.T = PCt_op_std.T
                        K_std = jax.scipy.linalg.solve(S_reg_op_std, PCt_op_std.T, assume_a='pos').T # assume_a='pos' hints that S is PSD
                        return K_std, jnp.all(jnp.isfinite(K_std))
                    except Exception:
                        return jnp.zeros_like(PCt_op_std.T).T, jnp.array(False)

                def _keep_previous_result_branch_local(operand_tuple_keep):
                    _, K_and_status_prev = operand_tuple_keep
                    return K_and_status_prev[0], K_and_status_prev[1]

                K_res_calc, current_solve_ok_calc = jax.lax.cond(
                    current_solve_ok_calc, # Predicate: True if Cholesky solve was successful
                    _keep_previous_result_branch_local, # Use Cholesky result
                    _standard_solve_branch_local,       # Otherwise, try standard solve
                    operand=((PCt_obs_val, S_obs_reg_val_calc), (K_res_calc, current_solve_ok_calc)) # Pass data and current status
                )

                # If standard solve also failed, attempt pinv (K = PCt @ pinv(S))
                def _pinv_solve_branch_local(operand_tuple_pinv):
                    matrices_for_pinv, _ = operand_tuple_pinv
                    PCt_op_pinv, S_reg_op_pinv = matrices_for_pinv
                    try:
                        K_pinv = PCt_op_pinv @ jnp.linalg.pinv(S_reg_op_pinv, rcond=1e-6) # Use a small rcond
                        return K_pinv, jnp.all(jnp.isfinite(K_pinv))
                    except Exception:
                        return jnp.zeros_like(PCt_op_pinv.T).T, jnp.array(False)

                K_res_calc, current_solve_ok_calc = jax.lax.cond(
                    current_solve_ok_calc, # Predicate: True if standard solve was successful
                    _keep_previous_result_branch_local, # Use standard solve result
                    _pinv_solve_branch_local,          # Otherwise, try pinv solve
                    operand=((PCt_obs_val, S_obs_reg_val_calc), (K_res_calc, current_solve_ok_calc)) # Pass data and current status
                )

                # Clip Kalman gain to prevent numerical blow-up
                K_res_calc = jnp.clip(K_res_calc, -1e3, 1e3)

                # Calculate log-likelihood contribution
                ll_t_calc = jnp.array(-1e6, dtype=P_pred.dtype) # Default if LL computation fails
                # Only compute LL if K gain solve was successful and S is valid
                def compute_ll_if_solve_ok(operand_ll):
                    S_reg_ll, v_ll, n_obs_ll = operand_ll
                    try:
                        sign, log_det_S = jnp.linalg.slogdet(S_reg_ll)
                        # Check if log_det_S is finite and sign is positive (for log)
                        valid_det = (sign > 0) & jnp.isfinite(log_det_S)

                        # Compute Mahalanobis distance: v' * S_reg_inv * v
                        # Can do this via Cholesky of S_reg: S_reg = L @ L.T
                        # v' @ (L @ L.T)^-1 @ v = v' @ L.T^-1 @ L^-1 @ v = (L^-1 @ v).T @ (L^-1 @ v)
                        # Solve L @ z = v for z. Mahalanobis distance is z.T @ z = sum(z^2).
                        L_S_ll = jnp.linalg.cholesky(S_reg_ll) # Assumes S_reg_ll is PSD (handled by regularization)
                        z = jax.scipy.linalg.solve_triangular(L_S_ll, v_ll, lower=True)
                        mah_d = jnp.sum(z**2)

                        # Check if mahalanobis distance is finite
                        valid_mah_d = jnp.isfinite(mah_d)

                        # LL formula: -0.5 * (n_obs_actual * log(2*pi) + log_det(S) + mahalanobis_distance)
                        log_pi_term = jnp.log(2 * jnp.pi) * n_obs_ll
                        ll_term = -0.5 * (log_pi_term + log_det_S + mah_d)

                        # Return LL term if all components are valid, otherwise return -1e6
                        return jnp.where(valid_det & valid_mah_d, ll_term, jnp.array(-1e6, dtype=P_pred.dtype))
                    except Exception:
                        # If Cholesky or other LL calculation fails, return -1e6
                        return jnp.array(-1e6, dtype=P_pred.dtype)

                ll_t_calc = jax.lax.cond(
                    current_solve_ok_calc, # Only compute LL if K gain solve was OK (implies S was likely valid)
                    compute_ll_if_solve_ok,
                    lambda op: jnp.array(-1e6, dtype=P_pred.dtype), # Return default if K solve failed
                    operand=(S_obs_reg_val_calc, v_obs_calc, static_n_obs_actual)
                )

                return K_res_calc, S_obs_reg_val_calc, ll_t_calc, current_solve_ok_calc

            # Perform update calculations only if perform_update is True
            K_res, S_obs_reg_val, ll_t, solve_status = jax.lax.cond(
                perform_update,
                do_update_calculations,
                # If not performing update, return default values for K, S, ll, and False status
                lambda operand: (
                    jnp.zeros((self.n_state, static_n_obs_actual if static_n_obs_actual > 0 else 0), dtype=x_pred_t.dtype),
                    static_I_obs * 1e6 if static_n_obs_actual > 0 else jnp.empty((0,0), dtype=ys_arr.dtype),
                    jnp.array(-1e6, dtype=I_s.dtype),
                    jnp.array(False)
                ),
                operand=(P_pred_t, static_C_obs, static_H_obs, static_I_obs, v_obs)
            )


            # Apply update using the calculated K, but only if solve was successful
            x_filt_t, P_filt_t = x_pred_t, P_pred_t # Initialize with predicted values

            # Define update functions for lax.cond
            def apply_update(operand_upd):
                K_, v_, x_, P_, C_, H_, I_s_ = operand_upd
                # Kalman update equations
                x_up = K_ @ v_
                x_f = x_ + x_up
                x_f = jnp.clip(x_f, -MAX_STATE_VALUE, MAX_STATE_VALUE) # Clip updated state

                IKC = I_s_ - K_ @ C_
                # Joseph form of covariance update for stability
                P_f = IKC @ P_ @ IKC.T + K_ @ H_ @ K_.T

                # Ensure P_f is symmetric and regularized
                P_f = (P_f + P_f.T) / 2.0
                P_f = P_f + kf_jitter * I_s_
                return x_f, P_f

            def skip_update(operand_skip):
                # If no update is performed, filtered state equals predicted state
                _, _, x_, P_, _, _, _ = operand_skip
                return x_, P_

            # Conditionally apply update based on whether solve was successful
            # (Note: perform_update ensures static_n_obs_actual > 0 and some finite data)
            x_filt_t, P_filt_t = jax.lax.cond(
                 solve_status, # Only update if K gain solve was successful
                 apply_update,
                 skip_update,
                 operand=(K_res, v_obs, x_pred_t, P_pred_t, static_C_obs, static_H_obs, I_s)
            )

            # --- Prepare outputs for this time step ---
            # Handle potential NaNs/Infs in outputs resulting from failed operations
            # If any NaN occurred during prediction/update, try to return a safe default
            # This might be redundant with clipping and conditional updates, but adds robustness
            x_pred_t_safe = jnp.where(jnp.isfinite(x_pred_t), x_pred_t, jnp.zeros_like(x_pred_t))
            P_pred_t_safe = jnp.where(jnp.all(jnp.isfinite(P_pred_t)), P_pred_t, I_s*1e6) # Use all() for matrices
            x_filt_t_safe = jnp.where(jnp.isfinite(x_filt_t), x_filt_t, jnp.zeros_like(x_filt_t))
            P_filt_t_safe = jnp.where(jnp.all(jnp.isfinite(P_filt_t)), P_filt_t, I_s*1e6)

            # Ensure covariance matrices are symmetric before returning
            P_pred_t_safe = (P_pred_t_safe + P_pred_t_safe.T) / 2.0
            P_filt_t_safe = (P_filt_t_safe + P_filt_t_safe.T) / 2.0

            # Innovation vector: v_obs is already calculated using cleaned y_obs_t
            # If no update was performed (perform_update is False), v_obs was calculated but LL is -1e6.
            # We should return a zero vector for innovations if perform_update was False,
            # as the innovation concept isn't meaningful when no update happens.
            innovations_out = jnp.where(perform_update, v_obs, jnp.zeros_like(v_obs))
             # Also handle case where v_obs might have NaNs if perform_update was true but y_obs_t_clean failed? No, nan_to_num prevents this.
            innovations_out = jnp.where(jnp.isfinite(innovations_out), innovations_out, jnp.zeros_like(innovations_out))

            # Innovation covariance: S_obs_reg_val is already calculated.
            # If no update was performed, S_obs_reg_val is the default large value.
            innovation_cov_out = jnp.where(perform_update, S_obs_reg_val, static_I_obs*1e6 if static_n_obs_actual>0 else jnp.empty((0,0),dtype=ys_arr.dtype))
             # Handle potential NaNs/Infs in S
            innovation_cov_out = jnp.where(jnp.all(jnp.isfinite(innovation_cov_out)), innovation_cov_out, static_I_obs*1e6 if static_n_obs_actual>0 else jnp.empty((0,0),dtype=ys_arr.dtype))
            innovation_cov_out = (innovation_cov_out + innovation_cov_out.T)/2.0 # Ensure symmetry

            # Log-likelihood contribution: ll_t is already calculated and defaulted to -1e6 on failure
            ll_t_safe = jnp.where(jnp.isfinite(ll_t), ll_t, jnp.array(-1e6, dtype=I_s.dtype))


            out = {
                'x_pred': x_pred_t_safe,
                'P_pred': P_pred_t_safe,
                'x_filt': x_filt_t_safe,
                'P_filt': P_filt_t_safe,
                'innovations': innovations_out, # Using potentially zeroed v_obs
                'innovation_cov': innovation_cov_out, # Using potentially default S
                'log_likelihood_contributions': ll_t_safe
            }

            # The next carry is the filtered state and covariance
            return (x_filt_t_safe, P_filt_t_safe), out

        # Run the scan operation over time steps
        init_carry = (self.init_x, self.init_P)
        # The scan outputs will be a dictionary where each value is a time-stacked array
        (_, _), scan_outputs = lax.scan(step_nan_handling, init_carry, ys_arr)

        # Final NaN/Inf check and symmetrization for covariance matrices in the final output arrays
        # (This might be redundant if handled correctly inside the loop, but adds safety)
        # Ensure P_pred and P_filt are symmetric and finite in the final stacked arrays
        if 'P_pred' in scan_outputs:
             P_pred_final = scan_outputs['P_pred']
             P_pred_final = (P_pred_final + jnp.transpose(P_pred_final, (0, 2, 1))) / 2.0
             scan_outputs['P_pred'] = jnp.where(jnp.all(jnp.isfinite(P_pred_final), axis=(-2,-1), keepdims=True), P_pred_final, self.I_s[None,:,:]*1e6)

        if 'P_filt' in scan_outputs:
             P_filt_final = scan_outputs['P_filt']
             P_filt_final = (P_filt_final + jnp.transpose(P_filt_final, (0, 2, 1))) / 2.0
             scan_outputs['P_filt'] = jnp.where(jnp.all(jnp.isfinite(P_filt_final), axis=(-2,-1), keepdims=True), P_filt_final, self.I_s[None,:,:]*1e6)

        if 'innovation_cov' in scan_outputs and static_n_obs_actual > 0:
             S_final = scan_outputs['innovation_cov']
             S_final = (S_final + jnp.transpose(S_final, (0, 2, 1))) / 2.0
             static_I_obs_3d = static_I_obs[None, :, :] # Add time dimension
             scan_outputs['innovation_cov'] = jnp.where(jnp.all(jnp.isfinite(S_final), axis=(-2,-1), keepdims=True), S_final, static_I_obs_3d*1e6)
        elif 'innovation_cov' in scan_outputs and static_n_obs_actual == 0:
             # Ensure innovation_cov is empty if n_obs_actual is 0
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

    # --- RTS Smoother (used internally or via smooth method) ---
    def _rts_smoother_backend(self, filter_results_dict: Dict) -> Tuple[jax.Array, jax.Array]:
        """
        Runs the Rauch-Tung-Striebel smoother on the filter results.
        Assumes filter_results_dict contains 'x_pred', 'P_pred', 'x_filt', 'P_filt'.
        Handles potential NaNs/Infs in filter outputs gracefully.
        """
        x_pred = filter_results_dict['x_pred'] # T x n_state
        P_pred = filter_results_dict['P_pred'] # T x n_state x n_state
        x_filt = filter_results_dict['x_filt'] # T x n_state
        P_filt = filter_results_dict['P_filt'] # T x n_state x n_state

        T_mat = self.T # n_state x n_state
        N = x_filt.shape[0] # Number of time steps

        # Handle empty time series case
        if N == 0:
            return jnp.empty((0, self.n_state), dtype=x_filt.dtype), jnp.empty((0, self.n_state, self.n_state), dtype=P_filt.dtype)

        kf_jitter_smooth_backend = _KF_JITTER
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


        # The smoothing loop runs backwards from N-2 down to 0 (inclusive)
        # The smoothed state and covariance at time N-1 are the filtered ones.
        # Let's structure the scan input and carry accordingly.
        # Carry: (x_s_next, P_s_next) - smoothed state/cov at time t+1
        # Scan input: data for time t+1 needed to compute J_t and update x_s_t, P_s_t
        # Specifically, we need: P_pred_{t+1}, P_filt_t, x_pred_{t+1}, x_filt_t

        # Smoothed state/cov at the last time step (N-1)
        x_s_N_minus_1 = x_filt_safe[N - 1]
        P_s_N_minus_1 = P_filt_safe[N - 1]

        # Prepare scan inputs for time steps 0 to N-2 (which correspond to indices 0 to N-2 in original arrays)
        # The loop runs from t = N-2 down to 0.
        # We need P_pred[t+1], P_filt[t], x_pred[t+1], x_filt[t] for each step t.
        # For t = N-2, we need P_pred[N-1], P_filt[N-2], x_pred[N-1], x_filt[N-2]
        # For t = 0, we need P_pred[1], P_filt[0], x_pred[1], x_filt[0]
        # The scan goes forwards in index, so we need to provide inputs in reverse order.
        # Inputs will be for t = N-2, N-3, ..., 0
        # Scan inputs: (P_pred[N-1], P_filt[N-2], x_pred[N-1], x_filt[N-2]), ..., (P_pred[1], P_filt[0], x_pred[1], x_filt[0])
        # These are slices from original arrays from index 1 to N-1 for pred, and 0 to N-2 for filt.
        # Then reversed.

        P_pred_for_scan = P_pred_safe[1:N]     # P_pred[1] ... P_pred[N-1]
        P_filt_for_scan = P_filt_safe[0:N - 1] # P_filt[0] ... P_filt[N-2]
        x_pred_for_scan = x_pred_safe[1:N]     # x_pred[1] ... x_pred[N-1]
        x_filt_for_scan = x_filt_safe[0:N - 1] # x_filt[0] ... x_filt[N-2]

        # Reverse the inputs for lax.scan
        scan_inputs = (
            P_pred_for_scan[::-1], # P_pred[N-1], ..., P_pred[1]
            P_filt_for_scan[::-1], # P_filt[N-2], ..., P_filt[0]
            x_pred_for_scan[::-1], # x_pred[N-1], ..., x_pred[1]
            x_filt_for_scan[::-1]  # x_filt[N-2], ..., x_filt[0]
        )

        def backward_step_common(carry_smooth, scan_t_rev_idx):
            # carry_smooth: (x_s_next_t, P_s_next_t) = smoothed state/cov at time t+1
            # scan_t_rev_idx: (Pp_next_t, Pf_t, xp_next_t, xf_t)
            #   where next = t+1, t is the current time step being smoothed (N-2 down to 0)
            #   Pp_next_t is P_pred_{t+1}, Pf_t is P_filt_t, xp_next_t is x_pred_{t+1}, xf_t is x_filt_t
            x_s_next_t, P_s_next_t = carry_smooth
            Pp_next_t, Pf_t, xp_next_t, xf_t = scan_t_rev_idx

            # --- Compute Smoother Gain J_t ---
            # J_t = P_filt_t @ T.T @ P_pred_{t+1}^-1
            # Numerically: Solve P_pred_{t+1}.T @ J_t.T = (T @ P_filt_t).T for J_t.T
            Pf_t_sym = (Pf_t + Pf_t.T) / 2.0 # Ensure symmetry
            Pp_next_reg_val = Pp_next_t + kf_jitter_smooth_backend * jnp.eye(self.n_state, dtype=Pp_next_t.dtype) # Regularize P_pred

            Jt_res = jnp.zeros((self.n_state, self.n_state), dtype=Pf_t.dtype)
            current_J_solve_ok = jnp.array(False)

            # Attempt 1: Solve using jax.scipy.linalg.solve (assumes Pp_next_reg_val is PSD)
            try:
                # Solves Pp_next_reg_val @ Jt_res.T = (T_mat @ Pf_t_sym).T
                # Using assume_a='pos' hints the solver that Pp_next_reg_val is positive definite
                Jt_T = jax.scipy.linalg.solve(Pp_next_reg_val, (T_mat @ Pf_t_sym).T, assume_a='pos')
                Jt_res = Jt_T.T
                current_J_solve_ok = jnp.all(jnp.isfinite(Jt_res))
            except Exception:
                current_J_solve_ok = jnp.array(False) # Solve failed

            # If standard solve failed, attempt pinv (Jt = Pf @ T.T @ pinv(Pp_next))
            def _fb_pinv_J_common_local(operand_tuple_J_pinv):
                # operand_tuple is ((T,Pf_sym,Pp_reg),(Jt_prev,Status_prev))
                matrices_for_pinv, _ = operand_tuple_J_pinv
                T_loc, Pf_s_loc, Pp_n_r_loc = matrices_for_pinv
                try:
                    # Calculate Pf @ T.T first, then multiply by pinv(Pp_next)
                    J_pinv = (Pf_s_loc @ T_loc.T) @ jnp.linalg.pinv(Pp_n_r_loc, rcond=1e-6) # Use a small rcond
                    return J_pinv, jnp.all(jnp.isfinite(J_pinv))
                except Exception:
                    return jnp.zeros_like(Pf_s_loc), jnp.array(False) # Return zero matrix if pinv fails

            def _keep_J_common_local(operand_tuple_J_keep):
                # operand_tuple is ((T,Pf_sym,Pp_reg),(Jt_prev,Status_prev))
                _, J_and_status_to_keep = operand_tuple_J_keep
                return J_and_status_to_keep[0], J_and_status_to_keep[1]

            # Conditionally select the result based on solve success
            Jt_res, current_J_solve_ok = jax.lax.cond(
                current_J_solve_ok, # Predicate: True if standard solve was OK
                _keep_J_common_local, # Use standard solve result
                _fb_pinv_J_common_local, # Otherwise, try pinv solve
                operand=((T_mat, Pf_t_sym, Pp_next_reg_val), (Jt_res, current_J_solve_ok)) # Pass data and current status
            )

            # Ensure Jt_res is finite; if not, set to zero (implies smoothing update will be identity)
            Jt_res_safe = jnp.where(jnp.isfinite(Jt_res), Jt_res, jnp.zeros_like(Jt_res))


            # --- Compute Smoothed State and Covariance ---
            # x_s_t = x_filt_t + J_t @ (x_s_{t+1} - x_pred_{t+1})
            # P_s_t = P_filt_t + J_t @ (P_s_{t+1} - P_pred_{t+1}) @ J_t.T

            # Calculate difference terms, handling potential NaNs in smoothed states from t+1
            x_d = x_s_next_t - xp_next_t # Difference in means
            P_d = P_s_next_t - Pp_next_t # Difference in covariances

            # Apply smoothing updates
            x_s_t = xf_t + Jt_res_safe @ x_d
            P_s_t = Pf_t_sym + Jt_res_safe @ P_d @ Jt_res_safe.T

            # Ensure P_s_t is symmetric and handle NaNs/Infs
            P_s_t = (P_s_t + P_s_t.T) / 2.0
            P_s_t = jnp.where(jnp.all(jnp.isfinite(P_s_t), axis=(-2,-1), keepdims=True), P_s_t, I_s * 1e6)

            # Handle NaNs/Infs in state mean
            x_s_t = jnp.where(jnp.isfinite(x_s_t), x_s_t, jnp.zeros_like(x_s_t))


            # The carry for the next step (t-1) is the smoothed state/cov at the current step (t)
            return (x_s_t, P_s_t), (x_s_t, P_s_t) # Return current smoothed state/cov as output

        # Initial carry for the backward scan: smoothed state/cov at time N-1
        init_carry_smooth = (x_s_N_minus_1, P_s_N_minus_1)

        # Run the backward scan
        # The outputs (x_s_rev, P_s_rev) will be the smoothed states/covariances
        # from time N-2 down to 0, stacked in reverse order (index 0 is time N-2, index N-2 is time 0).
        (_, _), (x_s_rev, P_s_rev) = lax.scan(backward_step_common, init_carry_smooth, scan_inputs)

        # Concatenate the smoothed states/covariances:
        # The scan output is for times N-2 down to 0 (in reverse order).
        # Prepend the smoothed state/cov at time N-1 (which was the initial carry).
        # Reverse the scan outputs to get time 0 to N-2 order.
        x_smooth = jnp.concatenate([x_s_rev[::-1], x_filt_safe[N - 1:N]], axis=0)
        P_smooth = jnp.concatenate([P_s_rev[::-1], P_filt_safe[N - 1:N]], axis=0)

        # Final safety check for NaNs/Infs and symmetry after concatenation
        x_smooth = jnp.where(jnp.isfinite(x_smooth), x_smooth, jnp.zeros_like(x_smooth))
        P_smooth = jnp.where(jnp.all(jnp.isfinite(P_smooth), axis=(-2,-1), keepdims=True), P_smooth, I_s[None,:,:] * 1e6)
        P_smooth = (P_smooth + jnp.transpose(P_smooth, (0, 2, 1))) / 2.0

        return x_smooth, P_smooth


    # --- Public methods ---
    def smooth(self,ys:ArrayLike,filter_results:Optional[Dict]=None,static_valid_obs_idx:Optional[jax.Array]=None,static_n_obs_actual:Optional[int]=None,static_C_obs_for_filter:Optional[jax.Array]=None,static_H_obs_for_filter:Optional[jax.Array]=None,static_I_obs_for_filter:Optional[jax.Array]=None)->Tuple[jax.Array,jax.Array]:
        """
        Runs the RTS smoother. Can take pre-computed filter results or compute them first.

        Args:
            ys: Observation data (required if filter_results is None).
            filter_results: Optional dictionary from a previous filter run.
            static_valid_obs_idx, static_n_obs_actual, etc.: Static observation info (required if filter_results is None).

        Returns:
            Tuple of (smoothed_state_means, smoothed_state_covariances).
        """
        ys_arr=jnp.asarray(ys,dtype=self.C.dtype) # Ensure dtype matches C matrix
        if filter_results is None:
            # Need static observation info to run the filter
            if (static_valid_obs_idx is None or static_n_obs_actual is None or static_C_obs_for_filter is None or static_H_obs_for_filter is None or static_I_obs_for_filter is None):
                raise ValueError("Static NaN info must be provided to smooth() if filter_results is None.")
            filter_outs_dict=self.filter(ys_arr,static_valid_obs_idx,static_n_obs_actual,static_C_obs_for_filter,static_H_obs_for_filter,static_I_obs_for_filter)
        else:
            filter_outs_dict=filter_results # Use provided filter results

        # Run the RTS smoother backend
        return self._rts_smoother_backend(filter_outs_dict)

    def log_likelihood(self, ys: ArrayLike, static_valid_obs_idx: jax.Array, static_n_obs_actual: int, static_C_obs: jax.Array, static_H_obs: jax.Array, static_I_obs: jax.Array) -> jax.Array:
        """
        Computes the total log-likelihood of the observations using the Kalman filter.
        This is the sum of the log_likelihood_contributions from the filter output.
        """
        filter_results = self.filter(ys, static_valid_obs_idx, static_n_obs_actual, static_C_obs, static_H_obs, static_I_obs)
        # Sum the log-likelihood contributions over time steps
        # Ensure contributions are finite before summing
        ll_contributions = jnp.where(jnp.isfinite(filter_results['log_likelihood_contributions']), filter_results['log_likelihood_contributions'], jnp.full_like(filter_results['log_likelihood_contributions'], -1e6))
        total_log_likelihood = jnp.sum(ll_contributions)
        return total_log_likelihood

# Standalone simulate_state_space function (not part of the class, but provided in original)
def _simulate_state_space_impl(P_aug: ArrayLike, R_aug: ArrayLike, Omega: ArrayLike, H_obs: ArrayLike, init_x: ArrayLike, init_P: ArrayLike, key: jax.random.PRNGKey, num_steps: int) -> Tuple[jax.Array, jax.Array]:
    """
    Simulates a state space model.

    Args:
        P_aug: State transition matrix (T in standard notation). Shape (n_state, n_state).
        R_aug: Shock impact matrix (R in standard notation). Shape (n_state, n_shocks).
        Omega: Observation matrix (C in standard notation). Shape (n_obs, n_state).
        H_obs: Observation noise covariance (H in standard notation). Shape (n_obs, n_obs).
        init_x: Initial state mean. Shape (n_state,).
        init_P: Initial state covariance. Shape (n_state, n_state).
        key: JAX random key.
        num_steps: Number of time steps to simulate.

    Returns:
        Tuple of (simulated_states, simulated_observations).
        Simulated states shape (num_steps, n_state).
        Simulated observations shape (num_steps, n_obs).
    """
    desired_dtype=jnp.result_type(P_aug, R_aug, Omega, H_obs, init_x, init_P) # Determine dtype from inputs
    P_aug_jax=jnp.asarray(P_aug,dtype=desired_dtype)
    R_aug_jax=jnp.asarray(R_aug,dtype=desired_dtype)
    Omega_jax=jnp.asarray(Omega,dtype=desired_dtype)
    H_obs_jax=jnp.asarray(H_obs,dtype=desired_dtype)
    init_x_jax=jnp.asarray(init_x,dtype=desired_dtype)
    init_P_jax=jnp.asarray(init_P,dtype=desired_dtype)

    n_aug=P_aug_jax.shape[0]
    n_aug_shocks=R_aug_jax.shape[1] if R_aug_jax.ndim==2 and R_aug_jax.shape[1]>0 else 0
    n_obs_sim=Omega_jax.shape[0]

    key_init,key_state_noise,key_obs_noise=random.split(key,3)
    kf_jitter_sim_impl=_KF_JITTER

    # Simulate initial state from N(init_x, init_P)
    # Add jitter to init_P for Cholesky stability
    try:
        init_P_reg = init_P_jax + kf_jitter_sim_impl * jnp.eye(n_aug,dtype=desired_dtype)
        L0 = jnp.linalg.cholesky(init_P_reg)
        z0 = random.normal(key_init, (n_aug,), dtype=desired_dtype)
        x0 = init_x_jax + L0 @ z0
    except Exception:
        # Fallback if Cholesky fails (e.g., init_P is not PSD even with jitter)
        # Just use the mean, or perhaps a larger jitter?
        x0 = init_x_jax # Using mean as fallback


    # Generate standard normal state shocks
    state_shocks_std_normal=random.normal(key_state_noise,(num_steps,n_aug_shocks),dtype=desired_dtype) if n_aug_shocks>0 else jnp.zeros((num_steps,0),dtype=desired_dtype)

    # Generate observation noise
    obs_noise_sim_val=jnp.zeros((num_steps,n_obs_sim),dtype=desired_dtype)
    if n_obs_sim>0:
        # Add jitter to H_obs for Cholesky stability
        try:
            H_obs_reg=H_obs_jax+kf_jitter_sim_impl*jnp.eye(n_obs_sim,dtype=desired_dtype)
            L_H_obs=jnp.linalg.cholesky(H_obs_reg)
            # Simulate eta ~ N(0, H_obs) via Cholesky: eta = L_H_obs @ z_eta
            z_eta=random.normal(key_obs_noise,(num_steps,n_obs_sim),dtype=desired_dtype)
            obs_noise_sim_val=z_eta@L_H_obs.T # Note: Chol(Sigma) @ Z.T = (Z @ Chol(Sigma).T).T
            # Z is (N, n_obs), L_H_obs is (n_obs, n_obs). Z @ L_H_obs.T gives (N, n_obs)
        except Exception:
             # Fallback to multivariate_normal if Cholesky fails. Use slightly more jitter for MVN stability?
            try:
                H_obs_reg_mvn=H_obs_jax+kf_jitter_sim_impl*10*jnp.eye(n_obs_sim,dtype=desired_dtype)
                obs_noise_sim_val=random.multivariate_normal(key_obs_noise,jnp.zeros(n_obs_sim,dtype=desired_dtype),H_obs_reg_mvn,shape=(num_steps,),dtype=desired_dtype)
            except Exception:
                 # Fallback if multivariate_normal also fails (very unlikely for PSD matrix with jitter)
                 # Return zeros, though this isn't simulating noise.
                 pass # obs_noise_sim_val remains zeros initialized above

    # Define the simulation step function for lax.scan
    def sim_step(x_prev,noise_t):
        # noise_t is a tuple (state_shock_std_t, obs_noise_t)
        eps_t,eta_t=noise_t

        # State equation: x_t = T x_{t-1} + R eps_t
        # R_aug @ eps_t gives the actual state shock
        shock_term=R_aug_jax@eps_t if n_aug_shocks>0 else jnp.zeros(n_aug,dtype=x_prev.dtype)
        x_curr=P_aug_jax@x_prev+shock_term # P_aug is T

        # Observation equation: y_t = C x_t + eta_t
        y_curr=Omega_jax@x_curr+eta_t if n_obs_sim>0 else jnp.empty((0,),dtype=x_curr.dtype) # Omega is C

        # Clip state values to prevent blow-up
        x_curr = jnp.clip(x_curr, -1e10, 1e10) # Use a large clip value

        return x_curr,(x_curr,y_curr)

    # Run the simulation scan
    # Scan takes (carry_init, xs) -> (carry_final, ys_out)
    # Here: (x0, (state_shocks_std_normal, obs_noise_sim_val)) -> (x_final, (states_sim_res, obs_sim_res))
    _,(states_sim_res,obs_sim_res)=lax.scan(sim_step,x0,(state_shocks_std_normal,obs_noise_sim_val))

    # Ensure outputs are finite (optional, but safe)
    states_sim_res = jnp.where(jnp.isfinite(states_sim_res), states_sim_res, jnp.zeros_like(states_sim_res))
    obs_sim_res = jnp.where(jnp.isfinite(obs_sim_res), obs_sim_res, jnp.zeros_like(obs_sim_res))


    return states_sim_res,obs_sim_res

# JIT compile the simulation function with static arg
simulate_state_space=jax.jit(_simulate_state_space_impl,static_argnames=('num_steps',))


# --- END OF FILE utils/Kalman_filter_jax.py ---