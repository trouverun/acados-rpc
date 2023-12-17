import subprocess
import config
from casadi import *
from scipy import interpolate

lbx_vec = [-np.inf, -np.inf, -np.inf, 1 / config.lin_vel_acc_scaler, -16.5 / config.lin_vel_acc_scaler, -np.inf, -config.steer_max, -config.throttle_min,
           0]
ubx_vec = [np.inf, np.inf, np.inf, 66 / config.lin_vel_acc_scaler, 16.5 / config.lin_vel_acc_scaler, np.inf, config.steer_max, config.throttle_max,
           np.inf]

lbu_vec = [-config.u_steer_max, -config.u_throttle_max, 0]
ubu_vec = [config.u_steer_max, config.u_throttle_max, 66 / config.lin_vel_acc_scaler]

lbg_vec = [0, -np.inf, -config.fslip_limit, -config.rslip_limit]
ubg_vec = [np.inf, 1, config.fslip_limit, config.rslip_limit]


import json
with open('bicycle_params.json', 'r') as infile:
    bicycle_params = json.load(infile)


class CasadiSolver:
    def __init__(self, midpoints):
        # Degree of interpolating polynomial
        self.polydeg = 3
        # Get collocation points
        tau_root = np.append(0, collocation_points(self.polydeg, 'legendre'))
        # Coefficients of the collocation equation
        C = np.zeros((self.polydeg + 1, self.polydeg + 1))
        # Coefficients of the continuity equation
        D = np.zeros(self.polydeg + 1)
        # Coefficients of the quadrature function
        B = np.zeros(self.polydeg + 1)

        # Construct polynomial basis
        for j in range(self.polydeg + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(self.polydeg + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0)
            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(self.polydeg + 1):
                C[j, r] = pder(tau_root[r])
            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        # Time horizon
        T = 8.
        h = 1 / 10
        self.N = int(T / h)  # number of control intervals

        # State
        x = MX.sym("x")
        y = MX.sym("y")
        theta = MX.sym("theta")
        omega = MX.sym("w")
        vx = MX.sym("vx")
        vy = MX.sym("vy")
        steer = MX.sym("steer")
        throttle = MX.sym("throttle")
        s = MX.sym("s")
        ocp_x = vertcat(x, y, theta, vx, vy, omega, steer, throttle, s)
        self.nx = ocp_x.size1()
        # Controls
        u_steer = MX.sym("usteer")
        u_throttle = MX.sym("uthrottle")
        u_s = MX.sym("us")
        ocp_u = vertcat(u_steer, u_throttle, u_s)
        self.nu = ocp_u.size1()

        # Track spline:
        cx_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 1], k=3)
        cy_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 2], k=3)
        c_s = midpoints[:, 0]
        c_x = midpoints[:, 1]
        c_y = midpoints[:, 2]
        c_w = midpoints[:, 3]
        c_dx = interpolate.splev(midpoints[:, 0], cx_spline, der=1)
        c_dy = interpolate.splev(midpoints[:, 0], cy_spline, der=1)
        cx_fun = interpolant("s_cx", "bspline", [c_s], c_x)
        # center y position
        cy_fun = interpolant("s_cy", "bspline", [c_s], c_y)
        # center dx:
        cdx_fun = interpolant("s_cdx", "bspline", [c_s], c_dx)
        # center dy:
        cdy_fun = interpolant("s_cdy", "bspline", [c_s], c_dy)
        # Center width:
        cw_fun = interpolant("s_cw", "bspline", [c_s], c_w)

        # Kinematic params
        car_mass = bicycle_params["car_mass"]
        car_max_steer = bicycle_params["car_max_steer"]
        car_lr = bicycle_params["car_lr"]
        car_lf = bicycle_params["car_lf"]
        car_inertia = bicycle_params["car_inertia"]
            # Drivetrain params
        car_Tm0 = bicycle_params["drivetrain.car_Tm0"]
        car_Tm1 = bicycle_params["drivetrain.car_Tm1"]
        car_Tr0 = bicycle_params["drivetrain.car_Tr0"]
        car_Tr1 = bicycle_params["drivetrain.car_Tr1"]
        car_Tr2 = bicycle_params["drivetrain.car_Tr2"]
            # Tire params
        wheel_Df = bicycle_params["wheel_Df"]
        wheel_Cf = bicycle_params["wheel_Cf"]
        wheel_Bf = bicycle_params["wheel_Bf"]
        wheel_Dr = bicycle_params["wheel_Dr"]
        wheel_Cr = bicycle_params["wheel_Cr"]
        wheel_Br = bicycle_params["wheel_Br"]

        # Model equations:
        Frx = (
                ((car_Tm0 + car_Tm1 * (vx * config.lin_vel_acc_scaler)) * throttle)
                - (car_Tr0 * (1 - tanh(car_Tr1 * (vx * config.lin_vel_acc_scaler)))
                   + car_Tr2 * (vx * config.lin_vel_acc_scaler) ** 2
                   )
        )
        af = atan2(vy + 0.1 / config.lin_vel_acc_scaler + (omega * car_lf / config.lin_vel_acc_scaler),
                   vx + 2.5 / config.lin_vel_acc_scaler) - steer * car_max_steer
        Ffy = wheel_Df * sin(wheel_Cf * arctan(wheel_Bf * af))
        ar = atan2(vy + 0.1 / config.lin_vel_acc_scaler - (omega * car_lr / config.lin_vel_acc_scaler),
                   vx + 2.5 / config.lin_vel_acc_scaler)
        Fry = wheel_Dr * sin(wheel_Cr * arctan(wheel_Br * ar))
        f_expr = vertcat(
            vx / (config.pos_scaler / config.lin_vel_acc_scaler) * cos(theta) - vy / (
                        config.pos_scaler / config.lin_vel_acc_scaler) * sin(theta),
            vx / (config.pos_scaler / config.lin_vel_acc_scaler) * sin(theta) + vy / (
                        config.pos_scaler / config.lin_vel_acc_scaler) * cos(theta),
            omega,
            (1 / car_mass * (
                        Frx - Ffy * sin(steer * car_max_steer))) / config.lin_vel_acc_scaler + vy * omega,
            (1 / car_mass * (
                        Fry + Ffy * cos(steer * car_max_steer))) / config.lin_vel_acc_scaler - vx * omega,
            1 / car_inertia * (Ffy * car_lf * cos(steer * car_max_steer) - Fry * car_lr),
            u_steer,
            u_throttle,
            u_s
        )
        f = Function('f', [ocp_x, ocp_u], [f_expr])

        psi = atan2(cdy_fun(s), cdx_fun(s))
        e_contour = sin(psi) * (x - cx_fun(s)) - cos(psi) * (y - cy_fun(s))
        e_lag = -cos(psi) * (x - cx_fun(s)) - sin(psi) * (y - cy_fun(s))
        l = Function("l", [ocp_x, ocp_u], [
            # 0.1*(config.pos_scaler * e_contour / 3) ** 2
            (config.pos_scaler * e_lag / 0.01) ** 2
            - (config.pos_scaler * u_s / 10) ** 2
            + 1*vx*steer + 1*vx*u_steer**2 + 0.1*u_throttle**2
        ])

        center_circle_deviation = (config.pos_scaler * x - config.pos_scaler * cx_fun(s)) ** 2 + (
                    config.pos_scaler * y - config.pos_scaler * cy_fun(s)) ** 2
        ax = (1 / car_mass * (Frx - Ffy * sin(steer * car_max_steer)))
        ay = (1 / car_mass * (Fry + Ffy * cos(steer * car_max_steer)))

        constraints_fun = Function("con_f", [ocp_x], [
            vertcat(
                (config.pos_scaler*cw_fun(s)) ** 2 - center_circle_deviation,
                ((ax+config.ax_shift) / config.max_ax) ** 2 + (ay / config.max_ay) ** 2,
                af,
                ar
            )
        ])

        # Start with an empty NLP
        w = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # For plotting x and u given w
        x_result = []
        u_result = []

        # "Lift" initial conditions
        Xk = MX.sym('X0', ocp_x.size1())
        w.append(Xk)
        lbw.append(np.zeros(ocp_x.size1()))
        ubw.append(np.zeros(ocp_x.size1()))
        x_result.append(Xk)

        # Formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k), ocp_u.size1())
            w.append(Uk)
            lbw.append(lbu_vec)
            ubw.append(ubu_vec)
            u_result.append(Uk)

            # State at collocation points
            Xc = []
            for j in range(self.polydeg):
                Xkj = MX.sym('X_' + str(k) + '_' + str(j), ocp_x.size1())
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append(lbx_vec)
                ubw.append(ubx_vec)

            # Loop over collocation points
            Xk_end = D[0] * Xk
            for j in range(1, self.polydeg + 1):
                xp = C[0, j] * Xk
                for r in range(self.polydeg):
                    xp = xp + C[r + 1, j] * Xc[r]
                # dynamics:
                fj = f(Xc[j - 1], Uk)
                g.append(h * fj - xp)
                lbg.append(np.zeros(ocp_x.size1()))
                ubg.append(np.zeros(ocp_x.size1()))
                # cost
                qj = l(Xc[j - 1], Uk)
                J = J + B[j] * qj * h
                # constrain:
                gj = constraints_fun(Xc[j - 1])
                g.append(gj)
                lbg.append(lbg_vec)
                ubg.append(ubg_vec)
                # propagate:
                Xk_end = Xk_end + D[j] * Xc[j - 1]

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k + 1), ocp_x.size1())
            w.append(Xk)
            lbw.append(lbx_vec)
            ubw.append(ubx_vec)
            x_result.append(Xk)

            # Add equality constraint
            g.append(Xk_end - Xk)
            lbg.append(np.zeros(ocp_x.size1()))
            ubg.append(np.zeros(ocp_x.size1()))

            # constrain:
            ge = constraints_fun(Xk_end)
            g.append(ge)
            lbg.append(lbg_vec)
            ubg.append(ubg_vec)

        # Concatenate vectors
        w = vertcat(*w)
        g = vertcat(*g)
        x_result = horzcat(*x_result)
        u_result = horzcat(*u_result)
        self.lbw = np.concatenate(lbw)
        self.ubw = np.concatenate(ubw)
        self.lbg = np.concatenate(lbg)
        self.ubg = np.concatenate(ubg)

        # Create an NLP solver
        prob = {'f': J, 'x': w, 'g': g}
        opts = {'ipopt.max_iter': 500, 'ipopt.print_level': 0}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        gen_opts = {}
        solver.generate_dependencies("nlp.c", gen_opts)
        subprocess.Popen("gcc -fPIC -shared -O1 nlp.c -o nlp.so", shell=True).wait()
        self.solver = nlpsol("solver", "ipopt", "./nlp.so", opts)

        # Function to get x and u trajectories from w
        self.trajectories = Function('trajectories', [w], [x_result, u_result], ['w'], ['x', 'u'])

    def solve(self, x0, u0=None):
        if len(x0.shape) == 1:
            x0 = np.tile(x0, [self.N+1, 1])

        if u0 is None:
            u0 = np.zeros([self.N, self.nu])

        w0 = [x0[0, :].T]

        for k in range(self.N):
            w0.append(u0[k, :].T)

            # State at collocation points
            for j in range(self.polydeg):
                w0.append(x0[k, :].T)

            w0.append(x0[k+1, :].T)

        w0 = np.concatenate(w0)

        self.lbw[:self.nx] = x0[0, :]
        self.ubw[:self.nx] = x0[0, :]

        sol = self.solver(x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        xstar, ustar = self.trajectories(sol["x"])
        return xstar, ustar, sol['f']


track = np.load("spain_gps.npy")

cx_spline = interpolate.splrep(track[:, 0], track[:, 1], k=3)
cy_spline = interpolate.splrep(track[:, 0], track[:, 2], k=3)
cw_spline = interpolate.splrep(track[:, 0], track[:, 3], k=3)

eval_positions = np.linspace(0, track[-1, 0], int(track[-1, 0]))
cx = interpolate.splev(eval_positions, cx_spline, der=0)
cy = interpolate.splev(eval_positions, cy_spline, der=0)
cdx = interpolate.splev(eval_positions, cx_spline, der=1)
cdy = interpolate.splev(eval_positions, cy_spline, der=1)
hdg = np.arctan2(cdy, cdx)
cw = interpolate.splev(eval_positions, cw_spline, der=0)

trajectory = []

x0 = np.array([
    cx[0] / config.pos_scaler,
    cy[0] / config.pos_scaler,
    hdg[0],
    60 / config.lin_vel_acc_scaler,
    0,
    0,
    0,
    0,
    0
])
u0 = None

track_copy = track.copy()[1:]
track_copy += [track[-1, 0], 0, 0, 0]
solver_track = np.r_[track, track_copy]
solver_wrapper = CasadiSolver(solver_track / config.pos_scaler)
n_iter = int(120 / (1/10))
prev_f = np.inf
for i in range(n_iter):
    if i % 10 == 0:
        print(f"----------------------------------------- Iteration {i}/{n_iter} ----------------------------------------------------------------")

    fstar = np.inf

    loops = 0
    threshold = 0.9
    while fstar > 0 or ((i != 0 and not (fstar / prev_f) >= threshold) and not loops > 15):
        xstar, ustar, fstar = solver_wrapper.solve(x0, u0)
        print(fstar)
        x0 = xstar.T
        u0 = ustar.T
        loops += 1
        if loops > 3:
            threshold = 0.9 - loops*0.025

    prev_f = fstar

    trajectory.append(x0.full()[0, [-1, 0, 1, 3]].flatten())
    if x0.full()[0, -1]*config.pos_scaler > int(track[-1, 0]) -10:
        break
    x0[:-1] = x0[1:]
    x0[-1] = x0[-1]
    u0[:-1] = u0[1:]
    u0[-1] = u0[-1]

trajectory = np.asarray(trajectory) * [config.pos_scaler, config.pos_scaler, config.pos_scaler, config.lin_vel_acc_scaler]

np.save("trajectory_test.npy", trajectory)