import scipy
from acados_template import AcadosOcp, AcadosOcpSolver
from solver.dynamic_bicycle import create_dynamic_bicycle_model
from scipy import interpolate
import numpy as np
import time
import config


class Solver:
    def __init__(self, N, sample_time, bicycle_params, midpoints, refpoints, gp, constraint_tightening):
        self.N = N
        self.sample_time = sample_time
        self.time_steps = np.linspace(sample_time, config.nonuniform_sample_low, N+1)
        tightening_region = len(self.time_steps[self.time_steps.cumsum() < 1.8])
        self.N_variance_propagation = tightening_region
        self.n_controls = 0
        self.n_states = 0
        self.n_parameters = 0
        self.gp = gp
        self.constraint_tightening = constraint_tightening

        # We need this to update the last track position after we shift the horizon and propagate the last stage with dynamics:
        self.cx_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 1], k=3)
        self.cy_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 2], k=3)
        self.cw_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 3], k=3)
        eval_positions = np.linspace(0, midpoints[-1, 0], int(config.pos_scaler * midpoints[-1, 0]))
        x = interpolate.splev(eval_positions, self.cx_spline, der=0)
        y = interpolate.splev(eval_positions, self.cy_spline, der=0)
        self.interpolated_track = np.c_[eval_positions, x, y]

        self.ocp = None
        self.acados_solver = None
        self.delay_comp_fun = None
        self.dynamics_jacobian_fun = None
        self._create_solver(bicycle_params, midpoints, refpoints)

        self.initialized = False
        self.x0 = np.zeros([self.N+1, self.n_states])
        self.u0 = np.zeros([self.N, self.n_controls])

        if self.gp is not None:
            self.jacobian_mu_fun = self.gp.get_mu_jacobian_fun().map(self.N_variance_propagation)

    def _create_solver(self, bicycle_params, midpoints, refpoints):
        self.ocp = AcadosOcp()
        self.ocp.dims.N = self.N

        self.delay_comp_fun, self.dynamics_jacobian_fun = create_dynamic_bicycle_model(
            self.ocp, bicycle_params, midpoints, refpoints, self.gp)
        self.dynamics_jacobian_fun = self.dynamics_jacobian_fun.map(self.N_variance_propagation)

        self.n_controls = self.ocp.model.u.size()[0]
        self.n_states = self.ocp.model.x.size()[0]
        self.n_parameters = self.ocp.model.p.size()[0]

        self.ocp.constraints.x0 = np.zeros(self.n_states)
        self.ocp.parameter_values = np.zeros(self.n_parameters)

        self.ocp.solver_options.tf = self.time_steps.sum()
        self.ocp.solver_options.time_steps = self.time_steps
        self.ocp.solver_options.integrator_type = "DISCRETE"

        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.qp_solver_warm_start = 2
        self.ocp.solver_options.hpipm_mode = 'SPEED'
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.regularize_method = "PROJECT"
        # self.ocp.solver_options.line_search_use_sufficient_descent = 1
        self.ocp.solver_options.levenberg_marquardt = 0.04 # 0.04
        self.ocp.solver_options.nlp_solver_step_length = 0.2
        # self.ocp.solver_options.qp_solver_ric_alg = 0
        # self.ocp.solver_options.qp_solver_iter_max = config.qp_solver_max_iter
        # self.ocp.solver_options.nlp_solver_max_iter = config.nlp_solver_max_iter
        self.ocp.solver_options.tol = config.solver_tolerance
        self.ocp.solver_options.print_level = 0

        # self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def delay_compensation(self, state, controls, dt):
        update = self.delay_comp_fun(state, controls, dt)
        update = np.asarray(update).flatten()
        updated_state = state.copy()
        updated_state += update

        car_pos = updated_state[:2]
        theta = updated_state[-1]
        distances = np.sqrt(np.sum(np.square(self.interpolated_track[:, 1:3] - car_pos), axis=1))
        i = np.argmin(distances)
        new_theta = self.interpolated_track[i, 0]
        # Detect lap change:
        if abs(new_theta - theta) > self.interpolated_track[-1, 0]/2:
            new_theta += self.interpolated_track[-1, 0]
        updated_state[-1] = new_theta

        return updated_state

    def initialize(self, initial_state, max_speed):
        self.acados_solver.set(0, "lbx", initial_state)
        self.acados_solver.set(0, "ubx", initial_state)

        dt = self.time_steps

        if self.initialized:
            # Detect lap change:
            if abs(initial_state[-1] - self.x0[0, -1]) > self.interpolated_track[-1, 0]/2:
                self.x0[:, -1] -= self.interpolated_track[-1, 0]
            self.x0[0] = initial_state
        else:
            initial_guess = np.r_[np.zeros(6), np.zeros(1), 0.25*np.ones(1), np.zeros(1)]
            self.x0 = np.tile(initial_state + initial_guess, (self.N+1, 1))

        # Variance propagation using previous solution:
        if self.gp is not None:
            track_constraint_w = interpolate.splev(self.x0[:, -1], self.cw_spline, der=0)
            sigma = np.zeros([self.N + 1, 6, 6])
            sigma[0, :3, :3] = np.diag([0.1/config.pos_scaler, 0.1/config.pos_scaler, np.deg2rad(1)])

            M = self.gp.n_inducing
            equally_spaced_time_values = np.linspace(dt[0], dt[-1], M)
            interp_func = scipy.interpolate.interp1d(dt, np.arange(len(dt)), kind='linear', bounds_error=False)
            inducing_indices = np.round(interp_func(equally_spaced_time_values)).astype(int)

            t1 = time.time_ns()
            gp_params, Kmm_list, Qm_list, alpha_list = self.gp.get_params(self.x0[inducing_indices, 3:8])

            pred_variances = np.zeros([self.N+1, 3])
            if None not in Kmm_list:
                pred_variances = self.gp.predict_covar(
                    self.x0[:, 3:8], self.x0[inducing_indices, 3:8], Kmm_list, Qm_list)
                jac_mu = self.jacobian_mu_fun(self.x0[:self.N_variance_propagation, 3:8].T, gp_params)
                jac_mu = np.asarray(jac_mu).reshape(3, -1, 5).swapaxes(0, 1)
                jac_mu = np.c_[np.zeros([self.N_variance_propagation, 3, 1]), jac_mu]
                jac_dynamics = self.dynamics_jacobian_fun(self.x0[:self.N_variance_propagation, 2:8].T)
                jac_dynamics = np.asarray(jac_dynamics).reshape(6, -1, 6).swapaxes(0, 1)

                B_d = np.r_[np.zeros([3, 3]), np.eye(3)]
                for i in range(1, self.N_variance_propagation):
                    f_hat_jac = np.eye(6) + dt[i] * jac_dynamics[i] + B_d @ (dt[i] * jac_mu[i])
                    sigma[i] = (B_d @ (dt[i]**2 * np.diag(pred_variances[i])) @ B_d.T + f_hat_jac @ sigma[i-1] @ f_hat_jac.T)
                sigma[self.N_variance_propagation:] += sigma[self.N_variance_propagation-1]
                #print(np.diagonal(sigma[-1]))
            t2 = time.time_ns()
            #print("GP stuff took %d ms" % ((t2-t1)/1e6))

        track_dx = interpolate.splev(self.x0[:, -1], self.cx_spline, der=1)
        track_dy = interpolate.splev(self.x0[:, -1], self.cy_spline, der=1)
        track_psi = np.arctan2(track_dy, track_dx)

        # Fill in x0, u0 and p for the solver:
        track_tighteners = []
        ci_val = scipy.stats.norm.ppf(0.95)
        for i in range(self.N + 1):
            if self.gp is not None:
                if self.constraint_tightening:
                    varvec = np.diagonal(sigma[i])
                    track_N = np.abs(np.array([np.sin(track_psi[i]), np.cos(track_psi[i])]))
                    track_tightener = min(
                        ci_val * np.sqrt((track_N @ varvec[:2]) * config.pos_scaler),
                        track_constraint_w[i]*config.pos_scaler - 1)

                    ax_tightener = ci_val * np.sqrt(pred_variances[i, 0]*config.lin_vel_acc_scaler)
                    ay_tightener = ci_val * np.sqrt(pred_variances[i, 1]*config.lin_vel_acc_scaler)
                else:
                    track_tightener = 0
                    ax_tightener = 0
                    ay_tightener = 0
                p = np.r_[
                    dt[i], track_tightener, ax_tightener, ay_tightener, gp_params
                ]
            else:
                track_tightener = 0
                p = np.r_[
                    dt[i], track_tightener, 0, 0
                ]

            track_tighteners.append(track_tightener)
            self.acados_solver.set(i, "p", p)

            if i > 0:
                ubx = self.ocp.constraints.ubx
                ubx[3] = max_speed
                self.acados_solver.set(i, "ubx", ubx)
            self.acados_solver.set(i, "x", self.x0[i])

            if i < self.N:
                ubu = self.ocp.constraints.ubu
                ubu[2] = max_speed
                self.acados_solver.set(i, "ubu", ubu)
                self.acados_solver.set(i, "u", self.u0[i])

        return np.asarray(track_tighteners)

    def _shift_horizon(self):
        self.x0[:-1] = self.x0[1:]
        self.u0[:-1] = self.u0[1:]
        # Update the last stage:
        self.x0[-1] = self.delay_compensation(self.x0[-1], np.zeros(3), config.nonuniform_sample_low)

    def solve(self):
        for i in range(4):
            status = self.acados_solver.solve()

        if status in [0, 2]:   # Success or timeout
            self.initialized = True
            for i in range(self.N+1):
                x = self.acados_solver.get(i, "x")
                self.x0[i] = x
                if i < self.N:
                    u = self.acados_solver.get(i, "u")
                    self.u0[i] = u
        else:
            print("STATUS", status)
            print(self.acados_solver.get_cost())
            print()

        state_horizon = self.x0.copy()
        control_horizon = self.u0.copy()
        self._shift_horizon()

        state_horizon[:, 6] *= -1
        control_horizon[:, 0] *= -1

        return state_horizon, control_horizon, status in [0, 2]

    def __del__(self):
        del self.gp