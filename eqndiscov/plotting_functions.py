"""Functions for plotting results during Lasso/SOCP optimization."""

import importlib
import warnings
import numpy as np
import numpy.linalg as la
import plotly.graph_objects as go
import plotly.io as pio

import eqndiscov.monomial_library_utils as mlu
import eqndiscov.L_curve_utils_lasso as lcu

importlib.reload(mlu)
importlib.reload(lcu)
pio.renderers.default = 'notebook+plotly_mimetype'
warnings.filterwarnings('ignore')


def plot_smooth_states(u1, u2, u_noise, u_actual):
    """Plot results of projection-based vs GP smoothing."""
    # Plot the results
    m = np.size(u1, 0)
    for i in range(m):
        fig = go.Figure(layout_title_text=f'u{i+1} results')
        fig.add_trace(
            go.Scatter(y=u_noise[i], name='Noisy data', mode='markers'))
        fig.add_trace(go.Scatter(y=u1[i], name='Projection method'))
        fig.add_trace(go.Scatter(y=u2[i], name='Gaussian Process'))
        fig.add_trace(go.Scatter(y=u_actual[i], name='Actual'))
        fig.update_layout(width=600, height=400)
        fig.show()

    # Plot the error
    for i in range(m):
        fig = go.Figure(layout_title_text=f'u{i+1} error results')
        fig.add_trace(
            go.Scatter(y=u1[i] - u_actual[i], name='Projection method'))
        fig.add_trace(
            go.Scatter(y=u2[i] - u_actual[i], name='Gaussian Process'))
        fig.update_layout(width=600, height=400)
        fig.show()


# SOCP RESULTS


def print_socp_residuals(u, u_noise, du, du_actual, A, nu):
    """Print residuals following SOCP optimization."""
    N = np.size(du) - 1         # Don't include initial condition
    C = 1 / np.sqrt(N)
    print('----Solution Residual ||A udot - u_proj||----')
    for i in range(2):
        res = C * la.norm(A @ du[i] - u[i])
        res_actual = C * la.norm(A @ du_actual[i] - u[i])
        print(f'Res{i}: {res:.8f}')
        print(f'Res{i} Actual: {res_actual:.8f}')

    # Integration residual (should equal sigma)
    print('-----Integration Residaul ||A udot - u_noise||-----')
    print(f'Expected (Sigma): {np.sqrt(nu)}')
    for i in range(2):
        res = C * la.norm(A @ du[i] - u_noise[i])
        res_actual = C * la.norm(A @ du_actual[i] - u_noise[i])
        print(f'Res{i}: {res:.8f}')
        print(f'Res{i}-Actual: {res_actual:.8f}')


def plot_states(u, u_noise, u_socp, u_actual):
    """Plot state estimates following SOCP optimization."""
    m = np.size(u, 0)
    for i in range(m):
        fig = go.Figure(layout_title_text=f'u{i+1} results')
        fig.add_trace(go.Scatter(y=u_noise[i], name='Noisy data'))
        fig.add_trace(go.Scatter(y=u[i], name='Before optimization'))
        fig.add_trace(go.Scatter(y=u_socp[i], name='After optimization'))
        fig.add_trace(go.Scatter(y=u_actual[i], name='Actual'))
        fig.update_layout(width=600, height=400)
        fig.show()


def plot_derivs(du_socp, du, du_actual, N_start, N_end):
    """Compare derivative results from SOCP and Tikreg."""
    m = np.size(du_socp, 0)
    for i in range(m):
        fig = go.Figure(layout_title_text=f'u{i+1}dot results')
        fig.add_trace(go.Scatter(y=du[i], name='TikReg'))
        fig.add_trace(go.Scatter(y=du_socp[i], name='SOCP'))
        fig.add_trace(go.Scatter(y=du_actual[i], name='Actual'))
        fig.update_layout(width=600, height=400)
        fig.show()

        fig = go.Figure(layout_title_text=f'u{i+1}dot error')
        fig.add_trace(go.Scatter(y=du[i] - du_actual[i], name='TikReg'))
        fig.add_trace(go.Scatter(y=du_socp[i] - du_actual[i], name='SOCP'))
        fig.update_layout(width=600, height=400)
        fig.show()
