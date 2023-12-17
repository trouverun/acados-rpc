gp_h = 180

KM_H = 1000/(60*60)

steer_max = 0.75
throttle_min = 0.9  # 0.9
throttle_max = 1  # 0.9
u_steer_max = 5  # 5
u_throttle_max = 10 # 10

# Weights for nonlinear lsq cost:
lag_weight = 10
contour_weight = 0.25

s_weight = 5
steer_weight = 2

u_steer_weight = 2
u_throttle_weight = 0.1
velocity_profile_weight = 1

fslip_limit = 0.4
rslip_limit = 0.125

# Soft constraint violation weights:
soft_vx_slack_weight = 5
soft_nl_track_circle_weight = 10
soft_nl_acceleration_weight = 10
soft_nl_slip_weight = 100
# solver params
qp_solver_max_iter = 200
nlp_solver_max_iter = 25
solver_tolerance = 1e-4

pos_scaler = 100
lin_vel_acc_scaler = 25

nonuniform_sample_low = 1/10

max_ay =   19
max_ax =   12
ax_shift = 4