# ---
# jupyter:
#     authors:
#         [name: Jacqui Wentz]
#     title: "Results"
#     jupytext:
#         cell_markers: '"""'
#     kernelspec:
#         display_name: Python 3
#         language: python
#         name: python3
# ---

# %% [markdown]
"""# Runs SOCP and IRW_Lasso to find coefficients/derivatives."""

# %% tags=["remove_input"]

import os
import numpy as np
import numpy.linalg as la
from scipy.integrate import solve_ivp
import scipy.special as sps
import pandas as pd
import plotly.graph_objects as go
import json
import warnings
import plotly.io as pio
import multiprocessing
import time
import matplotlib.pyplot as plt

import dsindy.monomial_library_utils as mlu
import dsindy.plotting_functions as pf
import dsindy.ODE_systems as odesys
import dsindy.utils as utils
import dsindy.optim_problems as op

# Import tools need to show plots in notebooks
pio.renderers.default = 'notebook+plotly_mimetype'
warnings.filterwarnings('ignore')

# %% tags=["remove_input"]

# Load arguments
bdir = '/home/jacqui/projects/DSINDy/'
with open(f'{bdir}/arguments.json', 'r') as fid:
    arguments = json.load(fid)

utils.disp_table(pd.DataFrame.from_dict(arguments,
                                        orient='index',
                                        columns=['values']),
                 dig=6)

nu = arguments['nu']
realization = arguments['realization']
system = arguments['system']
ttrain = arguments['ttrain']
N = arguments['N']
datadir = arguments['datadir']
get_GP = True
if N == 8000:
    get_GP = False

# %% tags=["remove_cell"]
# For Testing
bdir = '/home/jacqui/projects/DSINDy/'
nu = 1  # noise level (variance)
system = '5'
N = 2000  # number of samples
ttrain = 5  # training time
realization = 0
datadir = f'{bdir}/paper_noise_realizations/Lorenz_96/'
get_GP = True

# %% [markdown]
"""
# Set up the initial system with noise
"""

# %%
description = f'system={system}_N={N}_nu={nu}_realization={realization}'
sys_params, u0, d = odesys.get_system_values(system)
tend = ttrain * 2  # testing time (includes initial trianing)
if system == '5':
    tend = 20
tstep = 0.01  # step size for running ODE
m = np.size(u0)
p = int(sps.factorial(m + d) / (sps.factorial(m) * sps.factorial(d)))
perc_trun = 0.05  # leads to N * (1-2*perc_trun) training samples
N_start = int(N * perc_trun)
N_end = int(N * (1 - perc_trun))

# Time vector when measurements are taken
t = np.linspace(0, ttrain, num=N)

# Find actual measurements/derivative/coef vector
u, u_actual, du_actual, c_actual = odesys.setup_system(t,
                                                       nu,
                                                       d,
                                                       system[0],
                                                       sys_params=sys_params,
                                                       u0=u0)

# Replace noisy and smoothed data for given realization
data_fn = f'{datadir}/system={system}_ttrain={ttrain}_N={N}_nu={nu}.csv'
data = pd.read_csv(data_fn)
if get_GP:
    u, u_proj, u_smooth = utils.extract_data(data, realization, m)
else:
    u, u_proj = utils.extract_data(data, realization, m, get_GP=get_GP)

# %% [markdown]
"""
# Look at the errors of the smoothing and projection method
"""

# %%  tags=["remove_input"]
err_dict_u_prior = {}
err_dict_u_prior['noisy'] = utils.rel_err(u, u_actual)
if get_GP:
    err_dict_u_prior['GP'] = utils.rel_err(u_smooth, u_actual)
err_dict_u_prior['proj'] = utils.rel_err(u_proj, u_actual)

# Display results
columns = []
for i in range(np.size(u, 0)):
    columns.append(f'u{i+1} relative l1 error')
utils.disp_table(pd.DataFrame.from_dict(err_dict_u_prior,
                                        orient='index',
                                        columns=columns),
                 dig=6)

if get_GP:
    pf.plot_smooth_states(u_proj, u_smooth, u, u_actual)

# %%  tags=["remove_cell"]
#
# Plot of state+noise for presentation

cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Presentation figure
for i in range(m):
    plt.plot(t, u[i], '.', label='Measurements')
    plt.plot(t, u_actual[i], label='Actual', color=cols[3])
    plt.xlim([0, ttrain])
    # plt.ylim(-1.1, 1.5)
    plt.ylabel(fr'$u_{i+1}$')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.savefig(f'{bdir}/output/presentation_figs/{description}_u{i+1}.pdf')
    plt.show()

# %% [markdown]
"""
# Find derivatives using Tikhonov Regularization
"""

# %% tags=["remove_input"]

amax = 100
amin = 1e-12
if system == '5':
    amax = 100
    amin = 1e-16
startTime = time.time()
opt_params_deriv = {'tol': 1e-12, 'a_min': amin, 'a_max': amax}
du = np.zeros((m, N))
for i in range(m):
    du[i] = op.deriv_tik_reg(t,
                             u[i] - u[i, 0],
                             du_actual[i],
                             plot=True,
                             title=f'L curve for u{i+1} derivative',
                             opt_params=opt_params_deriv)

executionTime = (time.time() - startTime)
print('Execution time for Tikhonov Regularization: ' + str(executionTime))

# %% [markdown]
"""
# Define matrices/parameters needed for SOCP
"""

# %%

Theta_proj = mlu.make_Theta(u_proj, d=d)
W_proj = np.diag(1 / la.norm(Theta_proj, axis=0))
Theta_tilde = Theta_proj @ W_proj

# Use estimator for B
M2 = N * mlu.make_M2(u, u_proj, d, nu)
G2W_es = W_proj.T @ mlu.make_G2_from_moments(M2) @ W_proj
B_es = la.inv(G2W_es) @ Theta_tilde.T

# Use smoothed data for B
G2W_sm = Theta_tilde.T @ Theta_tilde
B_sm = la.inv(G2W_sm) @ Theta_tilde.T

# Integration matrix
A = utils.get_discrete_integral_matrix(t)
A_final = np.hstack((np.ones(N).reshape(-1, 1), A))

# Projection operation
Phi = A @ mlu.make_Theta(u_proj, d=d)
Phi = np.hstack((np.ones(N).reshape(-1, 1), Phi))
U_proj, sig_proj, Vh_proj = la.svd(Phi, full_matrices=False)
P_proj = U_proj @ U_proj.T

# Smoothing matrix
D = utils.get_derivative_matrix(t)

# Estimate smoothing parameter C and compare with actual value
Phi_pinv = Vh_proj.T @ np.diag(1 / sig_proj) @ U_proj.T
du_proj_T = Theta_proj @ ((Phi_pinv @ u_proj.T)[1:])
du_proj = du_proj_T.T
C_est = la.norm(D @ du_proj_T, axis=0) / np.sqrt(N)
C_actual = la.norm(D @ du_actual.T, axis=0) / np.sqrt(N)

# import eqndiscov.denoising_functions as df
#
# C_sub_vec = np.array([])
# C_sub_gp_vec = np.array([])
# for k in [80, 40, 20, 10, 5]:
#     du_proj_sub = du_proj[:, ::k]
#     t_sub = t[::k]
#     D_sub = utils.get_derivative_matrix(t_sub)
#     du_proj_smooth = df.smooth_data(t_sub, du_proj_sub)
#     C_est_sub = la.norm(
#         D_sub @ du_proj_sub.T, axis=0
#         ) / np.sqrt(np.size(du_proj_sub[0]))
#     C_est_sub_GP = la.norm(
#         D_sub @ du_proj_smooth.T, axis=0
#         ) / np.sqrt(np.size(du_proj_smooth[0]))
#     C_sub_vec = np.append(C_sub_vec, C_est_sub)
#     C_sub_gp_vec = np.append(C_sub_gp_vec, C_est_sub_GP)
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=[100, 200, 400, 800, 1600, 4000], y=C_sub_vec[::2], name='Subsample'))
# fig.add_trace(go.Scatter(
#     x=[100, 200, 400, 800, 1600, 4000], y=C_sub_gp_vec[::2],
#     name='Subsample+GP smooth'))
# fig.add_hline(y=C_est[0])
# fig.add_trace(go.Scatter(
#     x=[0, 2000], y=C_actual[0] * np.ones(2), name='Optimal'))
# fig.add_trace(go.Scatter(
#     x=[0, 2000], y=C_est[0] * np.ones(2), name='Original'))
# fig.update_yaxes(title_text='C/sqrt(N)', type='log')
# fig.update_xaxes(title_text='N_subsample')
# fig.update_layout(title_text='N=8000')
# fig.show()
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=[100, 200, 400, 800, 1600, 4000], y=C_sub_vec[1::2], name='Subsample'))
# fig.add_trace(go.Scatter(
#     x=[100, 200, 400, 800, 1600, 4000], y=C_sub_gp_vec[1::2],
#     name='Subsample+GP smooth'))
#
# fig.update_yaxes(title_text='C/sqrt(N)', type='log')
# fig.update_xaxes(title_text='N_subsample')
# fig.show()

print(f'C (estimated): {C_est}')
# print(f'C (subsampled to {np.size(du_proj_sub[0])}: {C_est_sub}')
# print(f'C (from Tikhonov regularization): {C_tikreg}")
print(f'Actual smoothness of du: {C_actual}')

# for i in range(m):
#     fig = go.Figure(layout_title_text=f'u{i+1}dot results')
#     fig.add_trace(go.Scatter(y=du_proj_sub[i], name='udottilde'))
#     fig.add_trace(go.Scatter(y=du_proj_smooth[i], name='Actual'))
#     fig.update_layout(width=600, height=400)
#     fig.show()

# %% [markdown]
"""
# Performing SOCP optimization
"""

# %%

startTime = time.time()

# u0_du_socp_es = np.zeros((m, N + 1))
u0_du_socp_sm = np.zeros((m, N + 1))
for i in range(m):
    print(f'Performing SOCP for species {i+1}:')
    sig_est = 1 / np.sqrt(N) * la.norm(u[i] - u_proj[i])
    a_exp = sig_est * np.sqrt((p + 1) / N)
    socp_opt_params = {
        'a_min': a_exp / 10,
        'a_max': a_exp * 10,
        'max_IRW_iter': 5
    }
    # u0_du_socp_es[i] = op.run_socp_optimization(
    #     P_proj @ u_proj[i], A_final, B_es, D, W_proj, C_est[i],
    #     opt_params=socp_opt_params)
    u0_du_socp_sm[i] = op.run_socp_optimization(P_proj @ u_proj[i],
                                                A_final,
                                                B_sm,
                                                D,
                                                W_proj,
                                                C_est[i],
                                                opt_params=socp_opt_params)

executionTime = (time.time() - startTime)
print('Execution time for IRW-SOCP: ' + str(executionTime))

# %% [markdown]
"""
# Run SOCP using theoretical value of gamma
"""

# %%
c_theory = np.zeros((m, p))
u0_du_socp_theory = np.zeros((m, N + 1))
for i in range(m):
    sig_guess = 1 / np.sqrt(N) * la.norm(u[i] - u_proj[i])
    alpha = sig_guess * np.sqrt((p + 1) / N)
    B_new = np.copy(B_sm)
    c_old = np.ones(p)
    for j in range(10):
        x = op.solve_socp(P_proj @ u_proj[i],
                          C_est[i],
                          A_final,
                          B_new,
                          D,
                          alpha,
                          checkResid=False)[0]
        cW = np.hstack((np.zeros(p).reshape(-1, 1), B_sm)) @ x
        coef_change = la.norm(W_proj @ cW - c_old) / la.norm(c_old)
        print(f'Change in coefs at iter {j+1}: {coef_change}')
        if coef_change < 1e-4:
            break
        c_old = W_proj @ cW
        # If entire coefficient vector is expected to be zero break
        if np.max(np.abs(cW)) < 1e-6:
            break
        Dc = np.diag(1 / (np.abs(cW) + 1e-4 * np.max(np.abs(cW))))
        B_new = Dc @ B_sm
    c_theory[i] = W_proj @ cW
    u0_du_socp_theory[i] = x

# %% [markdown]
"""
# Metrics after SOCP optimization: Derivative Error (truncated)
"""

# %% tags=["remove_input"]

# Pull out the derivative (note the first element is I.C.)
du_socp_sm = u0_du_socp_sm[:, 1:]
# du_socp_es = u0_du_socp_es[:, 1:]
du_socp_theory = u0_du_socp_theory[:, 1:]

# Pick which result to show plots for
du_socp = np.copy(du_socp_sm)
u0_du_socp = np.copy(u0_du_socp_sm)
u0_du_actual = np.hstack((np.array(u0).reshape(-1, 1), du_actual))
pf.plot_derivs(du_socp, du, du_actual, N_start, N_end)

# Truncate derivative
du_trun = du[:, N_start:N_end]
du_actual_trun = du_actual[:, N_start:N_end]
du_socp_sm_trun = du_socp_sm[:, N_start:N_end]
# du_socp_es_trun = du_socp_es[:, N_start:N_end]
du_socp_theory_trun = du_socp_theory[:, N_start:N_end]
err_dict_du = {}
err_dict_du['tikreg'] = utils.rel_err(du_trun, du_actual_trun)
err_dict_du['socp_sm'] = utils.rel_err(du_socp_sm_trun, du_actual_trun)
# err_dict_du['socp_es'] = utils.rel_err(du_socp_es_trun, du_actual_trun)
err_dict_du['socp_theory'] = utils.rel_err(du_socp_theory_trun, du_actual_trun)

# Display results
columns = []
for i in range(np.size(u, 0)):
    columns.append(f'Relative du{i+1} l2 error')
utils.disp_table(
    pd.DataFrame.from_dict(err_dict_du, orient='index', columns=columns))

# %% [markdown]
"""
# Learn coefficients the usual way with lasso

Here I'm only showing plots for the first iteration of lasso.
"""

# %%

amax_lasso = 100
amin_lasso = 1e-8
if system == '5' and nu == 1:
    amax_lasso = 1
    amin_lasso = 1e-6
lasso_opt_params = {
    'a_min': amin_lasso,
    'a_max': amax_lasso,
    'max_IRW_iter': 5,
    'max_iter': 100000,
    'tol': 1e-12
}

# With projection-smoothed dataset
Theta_proj_trun = mlu.make_Theta(u_proj[:, N_start:N_end], d=d)
W_proj_trun = np.diag(1 / la.norm(Theta_proj_trun, axis=0))
lasso_c = np.zeros((m, p))
for i in range(m):
    lasso_c[i] = op.run_weighted_lasso(Theta_proj_trun @ W_proj_trun,
                                       du_trun[i],
                                       W_proj_trun,
                                       type='Proj smoothed',
                                       species=f'u{i+1}',
                                       show_L_curve=True,
                                       opt_params=lasso_opt_params)

# # With GP dataset
# Theta_smooth_trun = mlu.make_Theta(u_smooth[:, N_start:N_end], d=d)
# W_smooth_trun = np.diag(1 / la.norm(Theta_smooth_trun, axis=0))
# lasso_c_GP = np.zeros((m, p))
# for i in range(m):
#     lasso_c_GP[i] = op.run_weighted_lasso(
#         Theta_smooth_trun @ W_smooth_trun, du_trun[i], W_smooth_trun,
#         type='GP smoothed', species=f'u{i+1}', show_L_curve=True,
#         opt_params=lasso_opt_params)

# %% [markdown]
"""
# Coefficient error comparison
"""
# %% tags=["remove_input"]

# Record coefficients
c_dict = {}
# c_dict['socp_es'] = (W_proj @ B_es @ du_socp_es.T).T
c_dict['socp_sm'] = (W_proj @ B_sm @ du_socp_sm.T).T
c_dict['socp_theory'] = c_theory
c_dict['lasso'] = lasso_c
# c_dict['lasso_GP'] = lasso_c_GP

# Calculate coefficient errors
c_err_dict = {}
for key, val in c_dict.items():
    c_err_dict[key] = utils.rel_err(val, c_actual)

# Display results
columns = []
for i in range(np.size(u, 0)):
    columns.append(f'Relative c{i+1} l2 error')
utils.disp_table(pd.DataFrame.from_dict(c_err_dict,
                                        orient='index',
                                        columns=columns),
                 dig=6)

# %% [markdown]
"""
# Plot prediction results
"""

# %% tags=["remove_input"]

t_test_temp = np.arange(0, tend * 1.0001, tstep)
idx_end_train = np.where(t_test_temp == ttrain)[0][0]

out = solve_ivp(odesys.run_monomial_ode, [0, t_test_temp[-1]],
                u0,
                args=[c_actual, d],
                t_eval=t_test_temp,
                rtol=1e-12,
                atol=1e-12)

# change IC
change_IC = True
if change_IC:
    u0_test = out.y[:, np.where(out.t == ttrain)[0][0]]
    u_actual_test = out.y[:, np.where(out.t == ttrain)[0][0]:]
    t_test_start = ttrain
    t_test = np.arange(t_test_start, tend + tstep / 2, tstep)
else:
    u0_test = np.copy(u0)
    u_actual_test = out.y
    t_test_start = 0
    t_test = np.copy(t_test_temp)


def run_ode(q, u0, c, d, t_test):
    """Function to run ode as separate process."""
    out = solve_ivp(odesys.run_monomial_ode, [t_test[0], t_test[-1]],
                    u0,
                    args=[c, d],
                    t_eval=t_test,
                    rtol=1e-12,
                    atol=1e-12)
    q.put(out)


sol_dict = {}
for key, val in c_dict.items():
    que = multiprocessing.Queue()
    pro = multiprocessing.Process(target=run_ode,
                                  name="Run_ODE",
                                  args=(que, u0_test, c_dict[key], d,
                                        t_test - t_test[0]))
    pro.start()

    runtime = 1000
    for time_iters in range(runtime + 1):
        # Check if queue has results, if not wait for a second
        if not que.empty():
            # print('getting results!')
            sol_dict[key] = que.get()
            break
        if time_iters == runtime:
            print(
                f'solve_ivp failed to return solution after {runtime} seconds')
            sol_dict[key] = 'Failed'
            break
        time.sleep(1)

    # Terminate foo
    pro.terminate()

    # Cleanup
    pro.join()

# Save relative L2 error
u_err_dict = {}
t_fail_dict = {}

fig1 = go.Figure()
fig2 = go.Figure()

for key, val in sol_dict.items():
    err_key = f'{key}'
    err_key_train = f'{key}_train'
    err_key_test = f'{key}_test'

    if sol_dict[key] == 'Failed':
        u_err_dict[err_key] = np.repeat(-1, m)
        u_err_dict[err_key_train] = np.repeat(-1, m)
        u_err_dict[err_key_test] = np.repeat(-1, m)
        t_fail_dict[err_key] = -1
    else:
        u_cur = val.y

        idx = 1
        u_err = utils.rel_err(u_cur[:, :idx], u_actual_test[:, :idx])
        while np.max(u_err) < .1:
            idx += 1
            if idx >= np.size(u_cur, 1) - 1:
                break
            u_err = utils.rel_err(u_cur[:, :idx], u_actual_test[:, :idx])
        t_fail_dict[err_key] = tstep * np.round(t_test[idx] / tstep)

        if np.size(u_cur, 1) == np.size(u_actual_test, 1):
            u_err_dict[err_key] = utils.rel_err(u_cur, u_actual_test)

            u_err_dict[err_key_train] = utils.rel_err(
                u_cur[:, :idx_end_train], u_actual_test[:, :idx_end_train])
            u_err_dict[err_key_test] = utils.rel_err(
                u_cur[:, idx_end_train:], u_actual_test[:, idx_end_train:])
        else:
            u_err_dict[err_key] = np.repeat(-1, m)
            u_err_dict[err_key_train] = np.repeat(-1, m)
            u_err_dict[err_key_test] = np.repeat(-1, m)

for i in range(m):
    fig1 = go.Figure()
    fig2 = go.Figure()
    for key, val in sol_dict.items():
        if sol_dict[key] == 'Failed':
            continue
        u_cur = val.y
        n = np.size(u_cur, 1)
        u_cur_err = u_cur - u_actual_test[:, :n]
        fig1.add_trace(go.Scatter(x=t_test, y=u_cur[i], name=key))
        fig2.add_trace(go.Scatter(x=t_test, y=u_cur_err[i], name=key))

    fig1.add_trace(go.Scatter(x=t_test, y=u_actual_test[i], name='Actual'))
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

# # Pull in values from weak sindy

# # Values for system=2b, N=1000, nu=0.01, realization=7
# c1_WSINDY = np.array(
#     [0, 0, 0.991728193300835, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# c2_WSINDY = np.array([
#     -0.336191693, 0.73528806, 0, 0, -0.601285429, 1.705789569, -2.603538184,
#     -0.46098231, -1.946309106, 0, 1.007476692, 0.850158343, 0, 0.63016903,
#     -1.219972351
# ])

# c_WSINDY = np.vstack((c1_WSINDY, c2_WSINDY))
# c_WSINDY.shape
# que = multiprocessing.Queue()
# pro = multiprocessing.Process(target=run_ode, name="Run_ODE",
#                               args=(que, tend, u0, c_WSINDY, d, t_test))
# pro.start()

# sol_dict['WSINDy'] = que.get()
# %% tags=["remove_cell"]

for i in range(m):
    plt.figure(figsize=(4, 2))
    for key, val in sol_dict.items():
        if key == 'socp_sm':
            label = 'DSINDy'
        if key == 'socp_theory':
            continue
        if key == 'lasso':
            label = r'$\ell_1$-SINDy'
        if key == 'WSINDy':
            label = 'WSINDy'
        if sol_dict[key] == 'Failed':
            continue
        u_cur = val.y
        plt.plot(t_test[:np.size(u_cur[i])], u_cur[i], label=label)
    plt.plot(t_test, u_actual_test[i], label='Actual', linestyle='dashed')
    plt.xlim([0, tend])
    # plt.ylim(-1.1, 1.5)
    # plt.ylim(-10, 10)
    plt.ylabel(fr'$u_{i+1}$')
    plt.xlabel(r'$t$')
    plt.legend(ncol=2)
    nm = f'{description}_u{i+1}_pred.pdf'
    plt.savefig(f'{bdir}/output/presentation_figs/{nm}')
    plt.show()
# %% tags=["remove_input"]

# for j in range(2):
#     fig1, axs1 = plt.subplots(2, 1)
#     fig2, axs2 = plt.subplots(2, 1)
#     fig1.set_size_inches(5, 3)
#     fig2.set_size_inches(5, 3)
#     if j == 0:
#         keys = ['socp_GP', 'lasso_GP']
#     if j == 1:
#         keys = ['socp_es', 'lasso']
#     for i in range(m):
#         for key, val in sol_dict.items():
#             if key not in keys:
#                 continue
#             u_cur = val.y
#             n = np.size(u_cur, 1)
#             u_cur_err = u_cur - u_actual_test[:, :n]
#             if i == 0:
#                 axs1[0].plot(t_test, u_cur[i], label=key)
#                 axs1[1].plot(t_test, u_cur_err[i], label=key)
#             else:
#                 axs2[0].plot(t_test, u_cur[i], label=key)
#                 axs2[1].plot(t_test, u_cur_err[i], label=key)
#
#         if i == 0:
#             axs1[0].plot(t_test, u_actual_test[i], label='Actual')
#         else:
#             axs2[0].plot(t_test, u_actual_test[i], label='Actual')
#
#     axs1[1].get_shared_x_axes().join(axs1[0], axs1[1])
#     axs2[1].get_shared_x_axes().join(axs2[0], axs2[1])
#     axs1[0].set_xticklabels([])
#     axs2[0].set_xticklabels([])
#     axs1[1].set_xlabel('Time')
#     axs2[1].set_xlabel('Time')
#     axs1[0].set_ylabel(r'$u_1$')
#     axs2[0].set_ylabel(r'$u_2$')
#     axs1[1].set_ylabel(r'$u_1$ error')
#     axs2[1].set_ylabel(r'$u_2$ error')
#     axs1[1].set_xlim(0, 20)
#     axs2[1].set_xlim(0, 20)
#     axs2[1].set_ylim(-5, 5)
#     axs1[0].set_ylim(-3, 3)
#
#     fig1.legend(['SOCP', 'Lasso', 'Actual'], ncol=3)
#     fig2.legend(['SOCP', 'Lasso', 'Actual'], ncol=3)
#
#     if j == 0:
#         fig1.savefig(f"/app/current_output/system={system}_GP_smoothing_u1_prediction_results_presentation.png")
#         fig2.savefig(f"/app/current_output/system={system}_GP_smoothing_u2_prediction_results_presentation.png")
#     if j == 1:
#         fig1.savefig(f"/app/current_output/system={system}_proj_smoothing_u1_prediction_results_presentation.png")
#         fig2.savefig(f"/app/current_output/system={system}_proj_smoothing_u2_prediction_results_presentation.png")
#
#     fig1.show()
#     fig2.show()

# %% tags=["remove_input"]
#
# i = 0
# for key, val in sol_dict.items():
#     u_cur = val.y
#     n = np.size(u_cur, 1)
#     u_cur_err = u_cur - u_actual_test[:, :n]
#     fig1.add_trace(go.Scatter(x=t_test, y=u_cur[i], name=key))
#     fig2.add_trace(go.Scatter(x=t_test, y=u_cur_err[i], name=key))
#
# fig, ax = plt.subplots()
# fig.set_size_inches(4, 2.5)
# plt.scatter(t, u[i], c='grey', s=1)
# plt.plot(t_test, sol_dict['socp_es'].y[i])
# plt.plot(t_test, sol_dict['lasso'].y[i])
# plt.plot(t_test, u_actual_test[i])
# ax.set_xlim([0, 30])
# ax.set_ylim([-3, 5])
# plt.legend(['Measurements', 'Predicted dynamics (our approach)',
#             'Predicted dynamics (alternative approach)', 'Actual dynamics'])
# plt.xlabel('Time')
# plt.ylabel('Predicted Dynamics')
# plt.savefig('current_output/VanDerPol_State1_Prediction.png')
# fig.show()

# %% [markdown] tags=["remove_input"]

# Examine predicton errors

# %% tags=["remove_input"]
# utils.disp_table(pd.DataFrame.from_dict(
#     u_err_dict, orient='index',
#     columns=['u1 prediction relative l2 error',
#              'u2 prediction relative l2 error']
# ), dig=6)
# %% [markdown] tags=["remove_input"]
"""
# Merge and save results
Merge all the error results and save into one large dataframe
"""
# %% tags=["remove_input"]


def append_value(data, string, i):
    """Return list of tuples with with label and value."""
    return [(string + k, v[i]) for k, v in data.items()]


def append_value_2(data, string):
    """Return list of tuples with with label and value."""
    return [(string + k, v) for k, v in data.items()]


summary_dict = {
    **dict([
        append_value(c_err_dict, f'c{i+1}_', i)[j] for j in range(
            len(c_err_dict)) for i in range(m)
    ]),
    **dict([
        append_value(u_err_dict, f'u{i+1}_pred_', i)[j] for j in range(
            len(u_err_dict)) for i in range(m)
    ]),
    **dict([
        append_value(err_dict_u_prior, f'u{i+1}_smooth_', i)[j] for j in range(
            len(err_dict_u_prior)) for i in range(m)
    ]),
    **dict([
        append_value(err_dict_du, f'du{i+1}_', i)[j] for j in range(
            len(err_dict_du)) for i in range(m)
    ]),
    **dict([
        append_value_2(t_fail_dict, 't_fail_')[j] for j in range(
            len(t_fail_dict))
    ])
}

summary_df = pd.DataFrame({
    'ErrType': list(summary_dict.keys()),
    'Value': list(summary_dict.values())
})
base_name = f'system={system}_nu={nu}_N={N}_ttrain={ttrain}'
out_dir = f'{bdir}/current_output/{base_name}'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
summary_df.to_csv(f'{out_dir}/{base_name}_realization={realization}.csv',
                  header=False,
                  index=False)

# %%
