from casadi import *
from acados_template import AcadosModel
import config
import scipy
from scipy import interpolate

fake_inf = 1E7

def create_dynamic_bicycle_model(ocp, bicycle_params, midpoints, refpoints, gp):
    ocp.model = AcadosModel()
    ocp.model.name = "dynamic_bicycle"

    # PARAMETERS ------------------------------------------------------------------------------------------------------

    # State
    x = MX.sym("x")
    y = MX.sym("y")
    theta = MX.sym("theta")
    vx = MX.sym("vx")
    vy = MX.sym("vy")
    w = MX.sym("w")
    steer = MX.sym("steer")
    throttle = MX.sym("throttle")
    s = MX.sym("s")
    ocp.model.x = vertcat(x, y, theta, vx, vy, w, steer, throttle, s)

    # Controls
    u_steer = MX.sym("usteer")
    u_throttle = MX.sym("uthrottle")
    u_s = MX.sym("us")
    ocp.model.u = vertcat(u_steer, u_throttle, u_s)

    # Configurable params
    dt = MX.sym("dt", 1)
    track_constraint_tightener = MX.sym("tct", 1)
    ax_tightener = MX.sym("axt", 1)
    ay_tightener = MX.sym("ayt", 1)

    copypoints = midpoints[:25].copy()
    additional_offset = np.sqrt(np.sum(np.square(midpoints[0, 1:3] - midpoints[-1, 1:3])))
    copypoints[:, 0] += midpoints[-1, 0] + additional_offset
    midpoints = np.r_[midpoints, copypoints]
    c_s = midpoints[:, 0]
    c_x = midpoints[:, 1]
    c_y = midpoints[:, 2]
    c_w = midpoints[:, 3]
    cx_fun = interpolant("s_cx", "bspline", [c_s], c_x)
    # center y position
    cy_fun = interpolant("s_cy", "bspline", [c_s], c_y)
    # Center width:
    cw_fun = interpolant("s_cw", "bspline", [c_s], c_w)

    copypoints = refpoints[:100].copy()
    additional_offset = np.sqrt(np.sum(np.square(refpoints[0, 1:3] - refpoints[-1, 1:3])))
    copypoints[:, 0] += refpoints[-1, 0] + additional_offset
    refpoints = np.r_[refpoints, copypoints]
    r_s = refpoints[:, 0]
    r_x = refpoints[:, 1]
    r_y = refpoints[:, 2]
    r_vx = refpoints[:, 3]
    rx_spline = interpolate.splrep(r_s, r_x, k=3)
    ry_spline = interpolate.splrep(r_s, r_y, k=3)
    r_dx = interpolate.splev(r_s, rx_spline, der=1)
    r_dy = interpolate.splev(r_s, ry_spline, der=1)
    rx_fun = interpolant("s_rx", "bspline", [r_s], r_x)
    ry_fun = interpolant("s_ry", "bspline", [r_s], r_y)
    rdx_fun = interpolant("s_drx", "bspline", [r_s], r_dx)
    rdy_fun = interpolant("s_dry", "bspline", [r_s], r_dy)
    rvx_fun = interpolant("s_rvx", "bspline", [r_s], r_vx)

    if gp is not None:
        residuals, residuals_params = gp.get_mu_symbolic_expression(ocp.model.x[3:-1])
        residuals = vertcat(0, 0, 0, residuals, 0, 0, 0)
        ocp.model.p = vertcat(dt, track_constraint_tightener, ax_tightener, ay_tightener, residuals_params)
    else:
        ocp.model.p = vertcat(dt, track_constraint_tightener, ax_tightener, ay_tightener)

    ocp.model.z = vertcat([])

    # DYNAMICS --------------------------------------------------------------------------------------------------------

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

    Frx = (
            ((car_Tm0 + car_Tm1 * (vx * config.lin_vel_acc_scaler)) * throttle)
            - (car_Tr0*(1-tanh(car_Tr1*(vx * config.lin_vel_acc_scaler))) + car_Tr2 * (vx * config.lin_vel_acc_scaler)**2)
    )

    af = atan2(vy+0.1/config.lin_vel_acc_scaler + (w*car_lf/config.lin_vel_acc_scaler), vx + 2.5/config.lin_vel_acc_scaler) - steer*car_max_steer
    Ffy = wheel_Df * sin(wheel_Cf * arctan(wheel_Bf * af))

    ar = atan2(vy+0.1/config.lin_vel_acc_scaler - (w*car_lr/config.lin_vel_acc_scaler), vx + 2.5/config.lin_vel_acc_scaler)
    Fry = wheel_Dr * sin(wheel_Cr * arctan(wheel_Br * ar))

    f_expr = vertcat(
        vx/(config.pos_scaler / config.lin_vel_acc_scaler) * cos(theta) - vy/(config.pos_scaler / config.lin_vel_acc_scaler) * sin(theta),
        vx/(config.pos_scaler / config.lin_vel_acc_scaler) * sin(theta) + vy/(config.pos_scaler / config.lin_vel_acc_scaler) * cos(theta),
        w,
        (1 / car_mass * (Frx - Ffy * sin(steer * car_max_steer))) / config.lin_vel_acc_scaler + vy*w,
        (1 / car_mass * (Fry + Ffy * cos(steer * car_max_steer))) / config.lin_vel_acc_scaler - vx*w,
        1 / car_inertia * (Ffy * car_lf * cos(steer * car_max_steer) - Fry * car_lr),
        u_steer,
        u_throttle,
        u_s
    )
    f = Function('f', [ocp.model.x, ocp.model.u], [f_expr])

    dyn_jac_state = vertcat(theta, vx, vy, w, steer, throttle)
    f_jac = Function("f_jac", [dyn_jac_state], [jacobian(f_expr[:6], dyn_jac_state)])

    # Discretize with RK4:
    k1 = f(ocp.model.x, ocp.model.u)
    k2 = f(ocp.model.x + dt / 2 * k1, ocp.model.u)
    k3 = f(ocp.model.x + dt / 2 * k2, ocp.model.u)
    k4 = f(ocp.model.x + dt * k3, ocp.model.u)
    dynamics = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    delay_compensation_f = Function("delay_comp", [ocp.model.x, ocp.model.u, dt], [dynamics])
    ocp.model.disc_dyn_expr = ocp.model.x + dynamics
    if gp is not None:
        ocp.model.disc_dyn_expr += dt * residuals

    # CONSTRAINTS -----------------------------------------------------------------------------------------------------

    # State bounds
    x_min = -fake_inf
    x_max = fake_inf
    y_min = -fake_inf
    y_max = fake_inf
    theta_min = -fake_inf
    theta_max = fake_inf
    w_min = -fake_inf
    w_max = fake_inf
    vx_min = 0
    vx_max = fake_inf
    vy_min = -50 / config.lin_vel_acc_scaler
    vy_max = 50 / config.lin_vel_acc_scaler
    steer_min = -config.steer_max
    steer_max = config.steer_max
    throttle_min = -config.throttle_min
    throttle_max = config.throttle_max
    s_min = 0
    s_max = fake_inf
    ocp.constraints.lbx = np.array([x_min, y_min, theta_min, vx_min, vy_min, w_min, steer_min, throttle_min, s_min])
    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx = np.array([x_max, y_max, theta_max, vx_max, vy_max, w_max, steer_max, throttle_max, s_max])
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx = np.arange(ocp.model.x.size()[0])
    ocp.constraints.idxbx_e = ocp.constraints.idxbx
    # Soft constraints:
    ocp.constraints.lsbx = np.array([0])
    ocp.constraints.lsbx_e = ocp.constraints.lsbx
    ocp.constraints.usbx = np.array([0])
    ocp.constraints.usbx_e = ocp.constraints.usbx
    ocp.constraints.idxsbx = np.array([3])
    ocp.constraints.idxsbx_e = ocp.constraints.idxsbx
    state_slack_weights = np.array([config.soft_vx_slack_weight])

    # Control bounds:
    u_steer_min = -config.u_steer_max
    u_steer_max = config.u_steer_max
    u_throttle_min = -config.u_throttle_max
    u_throttle_max = config.u_throttle_max
    u_s_min = 0
    u_s_max = fake_inf
    ocp.constraints.lbu = np.array([u_steer_min, u_throttle_min, u_s_min])
    ocp.constraints.ubu = np.array([u_steer_max, u_throttle_max, u_s_max])
    ocp.constraints.idxbu = np.arange(ocp.model.u.size()[0])
    # Soft constraints:
    ocp.constraints.lsbu = np.array([])
    ocp.constraints.usbu = np.array([])
    ocp.constraints.idxsbu = np.array([])

    # Nonlinear constraints
    ocp.constraints.lh = np.array([
        0,
        -fake_inf,
        -config.fslip_limit,
        -config.rslip_limit
    ])
    ocp.constraints.uh = np.array([
        fake_inf,
        1,
        config.fslip_limit,
        config.rslip_limit
    ])
    ocp.constraints.lh_e = np.array([
        0,
        0,
        -config.fslip_limit,
        -config.rslip_limit
    ])
    ocp.constraints.uh_e = np.array([
        fake_inf,
        0,
        config.fslip_limit,
        config.rslip_limit
    ])

    center_circle_deviation = (config.pos_scaler*x - config.pos_scaler*cx_fun(s))**2 + (config.pos_scaler*y - config.pos_scaler*cy_fun(s))**2

    ax = (1 / car_mass * (Frx - Ffy * sin(steer * car_max_steer)))
    ay = (1 / car_mass * (Fry + Ffy * cos(steer * car_max_steer)))
    if gp is not None:
        ax += config.lin_vel_acc_scaler * residuals[0]
        ay += config.lin_vel_acc_scaler * residuals[1]

    ocp.model.con_h_expr = vertcat(
        (config.pos_scaler*cw_fun(s) - track_constraint_tightener)**2 - center_circle_deviation,
        ((ax+config.ax_shift)/(config.max_ax - ax_tightener))**2 + (ay/(config.max_ay - ay_tightener))**2,
        af,
        ar
    )
    ocp.model.con_h_expr_e = vertcat(
        (config.pos_scaler*cw_fun(s) - track_constraint_tightener)**2 - center_circle_deviation,
        ay,
        af,
        ar
    )

    ocp.constraints.idxsh = np.array([
        0,
        1,
        2,
        3,
    ])
    ocp.constraints.idxsh_e = np.array([
        0,
        1,
        2,
        3
    ])
    nonlinear_slack_weights = np.array([
        config.soft_nl_track_circle_weight,
        config.soft_nl_acceleration_weight,
        config.soft_nl_slip_weight,
        config.soft_nl_slip_weight
    ])
    nonlinear_slack_weights_e = np.array([
        config.soft_nl_track_circle_weight,
        config.soft_nl_acceleration_weight,
        config.soft_nl_slip_weight,
        config.soft_nl_slip_weight
    ])

    # COSTS -----------------------------------------------------------1------------------------------------------------

    # Constraint hessian and diagonal
    ocp.cost.zl =   np.r_[10*state_slack_weights, nonlinear_slack_weights]
    ocp.cost.zl_e = np.r_[10*state_slack_weights, 2*nonlinear_slack_weights_e]
    ocp.cost.zu =   np.r_[state_slack_weights, nonlinear_slack_weights]
    ocp.cost.zu_e = np.r_[state_slack_weights, 2*nonlinear_slack_weights_e]

    ocp.cost.Zl =   np.r_[10*state_slack_weights, nonlinear_slack_weights]
    ocp.cost.Zl_e = np.r_[10*state_slack_weights, 2*nonlinear_slack_weights_e]
    ocp.cost.Zu =   np.r_[state_slack_weights, nonlinear_slack_weights]
    ocp.cost.Zu_e = np.r_[state_slack_weights, 2*nonlinear_slack_weights_e]

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    psi = atan2(rdy_fun(s), rdx_fun(s))
    e_contour = sin(psi) * (x - rx_fun(s)) - cos(psi) * (y - ry_fun(s))
    e_lag = -cos(psi) * (x - rx_fun(s)) - sin(psi) * (y - ry_fun(s))

    ocp.model.cost_y_expr = vertcat(
        (config.pos_scaler*e_contour)/1, (config.pos_scaler*e_lag)/0.1, 1+(config.pos_scaler*u_s)/180,
        vx*steer, vx*u_steer, u_throttle, config.lin_vel_acc_scaler*(vx - rvx_fun(s)) / 3
    )
    ocp.model.cost_y_expr_e = vertcat(
        (config.pos_scaler*e_contour)/1, (config.pos_scaler*e_lag)/0.1
    )
    ocp.cost.yref = np.array([0, 0, 0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0])

    ocp.cost.W = scipy.linalg.block_diag(
    config.contour_weight, config.lag_weight, -config.s_weight,
        config.steer_weight, config.u_steer_weight, config.u_throttle_weight, config.velocity_profile_weight
    )
    ocp.cost.W_e = config.nonuniform_sample_low*scipy.linalg.block_diag(
        config.contour_weight, config.lag_weight
    )

    return delay_compensation_f, f_jac
