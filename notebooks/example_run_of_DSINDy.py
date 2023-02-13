# %%

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import scipy.special as sps
import dsindy.ODE_systems as odesys
import dsindy.utils as utils
import dsindy.denoising_functions as df
import dsindy.optim_problems as op
import dsindy.monomial_library_utils as mlu
import dsindy.plotting_functions as pf

# %% User specified parameters

# Specify noise level and random seed
nu = 0.1
seed = 111111

# Specify system
# system = '2a'  # Duffing oscillator (PS1)
system = '2b'  # Duffing oscillator (PS2)
# system = '3'  # Van der Pol oscillator
# system = '4'  # Rossler attractor
# system = '5'  # Lorenz 96 model

# Set other system parameters
ttrain = 10  # Duration of training time
N = 1000  # Number of measurements

# Testing interval
t_test_start = 0
t_test_end = 10
tstep = 0.01

# %% Set up system
# Obtain system information and measurements
sys_params, u0, d = odesys.get_system_values(system)
m = np.size(u0)
p = int(sps.factorial(m + d) / (sps.factorial(m) * sps.factorial(d)))
t = np.linspace(0, ttrain, num=N)
u, u_actual, du_actual, c_actual = odesys.setup_system(t,
                                                       nu,
                                                       d,
                                                       system[0],
                                                       sys_params=sys_params,
                                                       u0=u0,
                                                       seed=seed)

# %% Perform iterPSDN
alpha = 0.1
max_iter = 1000
check_diverge = False
sigma_estimate = np.zeros(m)  # Not needed if check_diverge = False

# Projection-based smoothing
u_proj = np.empty((m, N))

# Projection-based smoothing
A = utils.get_discrete_integral_matrix(t)
u_proj = df.projection_denoising(u,
                                 u_actual,
                                 d,
                                 sigma_estimate,
                                 A,
                                 alpha=alpha,
                                 max_iter=max_iter,
                                 plot=True,
                                 use_actual_P=False,
                                 center_Theta=True,
                                 check_diverge=check_diverge)[0]

# Display relative l2 error for each state after denoising
err_dict_u_prior = {}
err_dict_u_prior['noisy'] = utils.rel_err(u, u_actual)
err_dict_u_prior['proj'] = utils.rel_err(u_proj, u_actual)
columns = []
for i in range(np.size(u, 0)):
    columns.append(f'u{i+1} relative l1 error')
utils.disp_table(pd.DataFrame.from_dict(err_dict_u_prior,
                                        orient='index',
                                        columns=columns),
                 dig=6)

# Plot results
pf.plot_smooth_states(u_proj, u, u_actual)

# %% IRW-SOCP

Theta_proj = mlu.make_Theta(u_proj, d=d)
W_proj = np.diag(1 / np.linalg.norm(Theta_proj, axis=0))
Theta_tilde = Theta_proj @ W_proj

# Use smoothed data for B
G2W = Theta_tilde.T @ Theta_tilde
B = np.linalg.inv(G2W) @ Theta_tilde.T

# Integration matrix
A = utils.get_discrete_integral_matrix(t)
A_final = np.hstack((np.ones(N).reshape(-1, 1), A))

# Projection operation
Phi = A @ mlu.make_Theta(u_proj, d=d)
Phi = np.hstack((np.ones(N).reshape(-1, 1), Phi))
U_proj, sig_proj, Vh_proj = np.linalg.svd(Phi, full_matrices=False)
P_proj = U_proj @ U_proj.T

# Smoothing matrix
D = utils.get_derivative_matrix(t)

# Estimate smoothing parameter C and compare with actual value
Phi_pinv = Vh_proj.T @ np.diag(1 / sig_proj) @ U_proj.T
du_proj_T = Theta_proj @ ((Phi_pinv @ u_proj.T)[1:])
du_proj = du_proj_T.T
C_est = np.linalg.norm(D @ du_proj_T, axis=0) / np.sqrt(N)
C_actual = np.linalg.norm(D @ du_actual.T, axis=0) / np.sqrt(N)
print(f'C (estimated): {C_est}')
print(f'Actual smoothness of du: {C_actual}')

u0_du_socp_sm = np.zeros((m, N + 1))
for i in range(m):
    print(f'Performing SOCP for species {i+1}:')
    sig_est = 1 / np.sqrt(N) * np.linalg.norm(u[i] - u_proj[i])
    a_exp = sig_est * np.sqrt((p + 1) / N)
    socp_opt_params = {
        'a_min': a_exp / 10,
        'a_max': a_exp * 10,
        'max_IRW_iter': 5
    }
    u0_du_socp_sm[i] = op.run_socp_optimization(P_proj @ u_proj[i],
                                                A_final,
                                                B,
                                                D,
                                                W_proj,
                                                C_est[i],
                                                opt_params=socp_opt_params)

# Pull out the derivative (note the first element is I.C.)
du_socp_sm = u0_du_socp_sm[:, 1:]

# Display derivative error results
err_dict_du = {}
err_dict_du['socp_sm'] = utils.rel_err(du_socp_sm, du_actual)
columns = []
for i in range(np.size(u, 0)):
    columns.append(f'Relative du{i+1} l2 error')
utils.disp_table(
    pd.DataFrame.from_dict(err_dict_du, orient='index', columns=columns))

# Calculate coefficients
c_dict = {}
c_dict['socp_sm'] = (W_proj @ B @ du_socp_sm.T).T

# Display coefficient error results
c_err_dict = {}
for key, val in c_dict.items():
    c_err_dict[key] = utils.rel_err(val, c_actual)
columns = []
for i in range(np.size(u, 0)):
    columns.append(f'Relative c{i+1} l2 error')
utils.disp_table(pd.DataFrame.from_dict(c_err_dict,
                                        orient='index',
                                        columns=columns),
                 dig=6)

# %% System prediction

# Calculate actual solution
t_test_temp = np.arange(0, t_test_end + tstep / 2, tstep)
idx_end_train = np.where(t_test_temp == t_test_start)[0][0]
out = solve_ivp(odesys.run_monomial_ode, [0, t_test_temp[-1]],
                u0,
                args=[c_actual, d],
                t_eval=t_test_temp,
                rtol=1e-12,
                atol=1e-12)

# If t_test different then 0 find new iniital conditions
if t_test_start > 0:
    u0_test = out.y[:, np.where(out.t == ttrain)[0][0]]
    u_actual_test = out.y[:, np.where(out.t == ttrain)[0][0]:]
    t_test_start = ttrain
    t_test = np.arange(t_test_start, t_test_end + tstep / 2, tstep)
else:
    u0_test = np.copy(u0)
    u_actual_test = out.y
    t_test_start = 0
    t_test = np.copy(t_test_temp)

# Run the ODE system
sol_dict = {}
for key, val in c_dict.items():
    sol_dict[key] = solve_ivp(odesys.run_monomial_ode, [t_test[0], t_test[-1]],
                              u0_test,
                              args=[c_dict[key], d],
                              t_eval=t_test,
                              rtol=1e-12,
                              atol=1e-12)

# Find relative l2 error
u_err_dict = {}
t_fail_dict = {}
for key, val in sol_dict.items():
    err_key = f'{key}'
    u_cur = val.y

    # Find 'time of failure'
    idx = 2
    u_err = utils.rel_err(u_cur[:, :idx], u_actual_test[:, :idx])
    while np.max(u_err) < .1:
        idx += 1
        if idx >= np.size(u_cur, 1) - 1:
            break
        u_err = utils.rel_err(u_cur[:, :idx], u_actual_test[:, :idx])
    t_fail_dict[err_key] = tstep * np.round(t_test[idx - 2] / tstep)

    if np.size(u_cur, 1) == np.size(u_actual_test, 1):
        u_err_dict[err_key] = utils.rel_err(u_cur, u_actual_test)
    else:
        u_err_dict[err_key] = np.repeat(-1, m)

# Display error
columns = []
for i in range(np.size(u, 0)):
    columns.append(f'Relative u{i+1} l2 error')
utils.disp_table(pd.DataFrame.from_dict(u_err_dict,
                                        orient='index',
                                        columns=columns),
                 dig=6)

# Plot results
for i in range(m):
    fig1 = go.Figure()
    fig2 = go.Figure()
    for key, val in sol_dict.items():
        u_cur = val.y
        n = np.size(u_cur, 1)
        u_cur_err = u_cur - u_actual_test[:, :n]
        fig1.add_trace(go.Scatter(x=t_test, y=u_cur[i], name=key))
        fig2.add_trace(go.Scatter(x=t_test, y=u_cur_err[i], name=key))
    fig1.add_trace(go.Scatter(x=t_test, y=u_actual_test[i], name='Actual'))
    fig1.add_vline(x=t_fail_dict['socp_sm'], line_dash='dash')
    fig2.add_vline(x=t_fail_dict['socp_sm'], line_dash='dash')
    fig1.update_layout(title_text=f'Simulation Results (u{i+1})',
                       width=600,
                       height=400)
    fig1.update_xaxes(title_text='Time')
    fig1.update_yaxes(title_text=f'u{i+1}')
    fig1.show()

    fig2.update_layout(title_text=f'Prediction Error (u{i+1})',
                       width=600,
                       height=400)
    fig2.update_xaxes(title_text='Time')
    fig2.update_yaxes(title_text=f'u{i+1}')
    fig2.show()

# %%
