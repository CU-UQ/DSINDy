"""Contains functions for smoothing/denoising state measurements."""
import numpy as np
import numpy.linalg as la
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import plotly.graph_objects as go
import plotly.io as pio
import warnings
pio.renderers.default = 'notebook+plotly_mimetype'
warnings.filterwarnings('ignore')

import eqndiscov.monomial_library_utils as mlu
import eqndiscov.utils as utils


def smooth_data(t, u):
    """Return data smoothed with Gaussian Process regression."""
    for i, ui in enumerate(u):
        kernel = RBF() + WhiteKernel()
        gp_fit = GPR(kernel=kernel, n_restarts_optimizer=10)
        gp_fit.fit(t[..., None], ui[..., None])
        u_smooth_temp, u_stdev = gp_fit.predict(t[..., None], return_std=True)
        u_smooth_fit = u_smooth_temp.flatten()
        if i == 0:
            u_smooth = np.copy(u_smooth_fit.reshape(1, -1))
        else:
            u_smooth = np.vstack((u_smooth, u_smooth_fit.reshape(1, -1)))

    return(u_smooth)


def projection_denoising(u, u_actual, d, sigma_estimate, A, max_iter=10,
                         plot=True, alpha=.1, use_actual_P=False,
                         center_Theta=False, check_diverge=True):
    """Perform projection-based denoising.

    Args:
        u (d X N np.array): description
        u_actual (d X N np.array): description
        d (int):
        sigma_estimate (d np.array):
        A ():
        max_iter (int):
        plot (bool):
        alpha (float in [0,1]):

    Returns:
        type: description


    """
    N = np.size(u, 1)
    m = np.size(u, 0)
    u_proj = np.copy(u)
    u_err_vec = []
    sigma_vec = []
    sum_vec = []
    u_norm = la.norm(u_actual, axis=1)

    for i in range(max_iter):

        # Record error history
        u_err_vec.append(
            la.norm(u_proj - u_actual, axis=1) / u_norm)

        # Make Phi which is A\Theta and a column of ones
        if use_actual_P is True:
            Theta_temp = mlu.make_Theta(u_actual, d=d)
        else:
            Theta_temp = mlu.make_Theta(u_proj, d=d)
            if center_Theta:
                Theta_temp = mlu.center_Theta(
                    Theta_temp, d, m, sigma_estimate[0]**2)

        Phi = A @ Theta_temp
        if d == 0:
            Phi = Phi.reshape(-1, 1)
        Phi = np.hstack((np.ones(N).reshape(-1, 1), Phi))

        # Use SVD to perform projeciton
        U, s, Vh = la.svd(Phi, full_matrices=False)
        P_Phi = U @ U.T
        u_proj_new = alpha * (P_Phi @ u_proj.T).T + (1 - alpha) * u_proj

        # # Use SOCP do find projection
        # B = alpha * P_Phi + (1 - alpha) * np.eye(np.size(u_proj, 1))
        # for i, u_proj_i in enumerate(u_proj):
        #     if i == 0:
        #         u_proj_new = solve_socp(u_proj_i, B)
        #     else:
        #         u_proj_new = np.vstack((u_proj_new, solve_socp(u_proj_i, B)))

        # Record mean of error
        sum_vec.append(1 / np.sqrt(N) * np.sum(u_proj_new - u, axis=1))

        # Record variance history
        sigma_pred = 1 / np.sqrt(N) * la.norm(u_proj_new - u, axis=1)
        sigma_vec.append(sigma_pred)

        # Check for divergence and break if sigma_pred is too large
        if check_diverge:
            if use_actual_P is not True and max_iter > 1:
                update = (sigma_pred < sigma_estimate)
                if sum(update) == 0:
                    print('WARNING: HIT MAX SIGMA')
                    break
                # If varaince too large don't perform projection in that direction
                u_proj_new[update == 0, :] = u_proj[update == 0, :]

        if np.max(utils.rel_err(u_proj_new, u_proj)) < 1e-8:
            print('Converged.')
            break

        u_proj = np.copy(u_proj_new)

    # Add final error to history
    u_err_vec.append(la.norm(u_proj - u_actual, axis=1) / u_norm)

    u_err_vec = np.array(u_err_vec).T
    sigma_vec = np.array(sigma_vec).T
    sum_vec = np.array(sum_vec).T

    if plot:
        fig = go.Figure()
        for i, ui in enumerate(u_err_vec):
            fig.add_trace(go.Scatter(y=ui, name=f'u{i} error'))
        fig.update_yaxes(title_text='Relative l2 error', type='log')
        fig.update_xaxes(title_text='Iteration of projection method')
        fig.update_layout(width=600, height=400)
        fig.show()

        fig = go.Figure()
        for i, ui in enumerate(sigma_vec):
            fig.add_trace(go.Scatter(y=ui, name=f'u{i}'))
        fig.update_yaxes(title_text='Sigma', type='log')
        fig.update_xaxes(title_text='Iteration of projection method')
        fig.update_layout(width=600, height=400)
        fig.show()

        fig = go.Figure()
        for i, ui in enumerate(sum_vec):
            fig.add_trace(go.Scatter(y=np.abs(ui), name=f'u{i}'))
        fig.update_yaxes(title_text='Error sum', type='log')
        fig.update_xaxes(title_text='Iteration of projection method')
        fig.update_layout(width=600, height=400)
        fig.show()

    return u_proj, la.cond(Phi)


from cvxopt import matrix, solvers


def solve_socp(u, B):
    """Solve quadratic program."""

    n = np.size(u)

    PP = matrix(np.eye(n))
    q = matrix(-B @ u)

    G = matrix(0.0, (n, n))
    h = matrix(0.0, (n, 1))

    A = matrix(1.0, (1, n))
    b = matrix(np.sum(u))

    sol = solvers.qp(PP, q, G, h, A, b)

    x = np.array(sol['x']).flatten()

    return(x)
