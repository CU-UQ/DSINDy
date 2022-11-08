"""Additional functions need for quick calculates and displaying results."""

from IPython.display import display
from functools import reduce
import operator as op
import numpy as np
import numpy.linalg as la


def ncr(n, r):
    """Calculate n choose r."""
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def disp_table(df, dig=4):
    """Print disp_output string to html file."""
    display(round(df, dig))


def rel_err(x, xtrue, axis=1):
    """Calculate the relative error w.r.t. xtrue."""
    n = np.min((np.size(x, 1), np.size(xtrue, 1)))
    true_norm = la.norm(xtrue[:, :n], axis=axis)
    return la.norm(x[:, :n] - xtrue[:, :n], axis=axis) / true_norm


def extract_data(data, j, m, get_GP=True):
    """Extract noisy, projection-smooth, and GP-smoothed data."""
    for i in range(m):
        ui = data[f'S{j+1}-noise-u{i+1}']
        if get_GP:
            ui_gp = data[f'S{j+1}-gp-u{i+1}']
        ui_proj = data[f'S{j+1}-proj-u{i+1}']
        if i == 0:
            u = ui
            u_proj = ui_proj
            if get_GP:
                u_gp = ui_gp
        else:
            u = np.vstack((u, ui))
            u_proj = np.vstack((u_proj, ui_proj))
            if get_GP:
                u_gp = np.vstack((u_gp, ui_gp))
    if get_GP:
        return(u, u_proj, u_gp)
    else:
        return(u, u_proj)

def get_discrete_integral_matrix(t, center=True):
    """Generative the discrete integration matrix."""
    n = np.size(t)
    td = t[1] - t[0]
    A = np.tril(td * np.ones((n, n)), 0)
    if center:
        for i in range(n):
            A[i, i] = A[i, i] / 2
        A[:, 0] = A[:, 0] - A[0, 0]
    return(A)


def get_derivative_matrix(t):
    """Generatives the differential matrix D."""
    t_delta = t[1] - t[0]
    m = np.size(t)

    # First order differential operator
    D1 = np.zeros((m - 1, m))
    D1[0:m - 1, 0:m - 1] = -np.eye(m - 1)
    D1[0:m - 1, 1:m] = D1[0:m - 1, 1:m] + np.eye(m - 1)

    # Second order differential operator
    D2 = np.zeros((m - 2, m))
    D2[0:m - 2, 1:m - 1] = -2 * np.eye(m - 2)
    D2[0:m - 2, 0:m - 2] = D2[0:m - 2, 0:m - 2] + np.eye(m - 2)
    D2[0:m - 2, 2:m] = D2[0:m - 2, 2:m] + np.eye(m - 2)

    D = np.vstack((np.eye(m), 1 / t_delta * D1, 1 / (t_delta**2) * D2))

    return(D)
