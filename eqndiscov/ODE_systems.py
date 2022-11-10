"""Functions needed to set up and run ODE systems."""

import numpy as np
import scipy.special as sps
from scipy.integrate import solve_ivp

import eqndiscov.monomial_library_utils as mlu


def get_coefficient_vector(system, sys_params, m, p):
    """Get coefficinet vector for a given system."""
    c = np.zeros((m, p))
    if system == '1':
        c[0, 2] = 1
        c[1, 1] = -np.pi**2
    if system == '2':
        gamma = sys_params['gamma']
        kappa = sys_params['kappa']
        epsilon = sys_params['epsilon']
        c[0, 2] = 1
        c[1, 1] = -kappa
        c[1, 2] = -gamma
        c[1, 6] = -epsilon
    if system == '3':
        gamma = sys_params['gamma']
        c[0, 2] = 1
        c[1, 1] = -1
        c[1, 2] = gamma
        c[1, 7] = -gamma
    if system == '4':
        alpha = sys_params['alpha']
        beta = sys_params['beta']
        kappa = sys_params['kappa']
        c[0, 2] = -1
        c[0, 3] = -1
        c[1, 1] = 1
        c[1, 2] = alpha
        c[2, 0] = beta
        c[2, 3] = -kappa
        c[2, 6] = 1
    return (c)


def setup_system(t, nu, d, system, sys_params=None, u0=[0, 1], seed=123456):
    """Generate the noisy state measurements for a given system.

    Args:
        t (np.array): time points at which to return system values
        nu (float): noise variance
        p (int): max polynomial order
        system (str):
            "1": Harmonic Oscillato";
            "2a" or "2b": Durring Oscillator;
            "3a" or "3b": Van der Pol Oscillator;
            "4": Rossler system
        u0 (list): initial conditions
        seed (int): seed for random number generator

    Returns:
        np.array (dxN): noisy state measurements
        np.array (dxN): actual state measurements
        np.array (dxN): actual state derivative
        np.array (dxP): actual coefficient vector
    """
    # Set parameter values
    N = np.size(t)
    m = np.size(u0)
    p = int(sps.factorial(m + d) / (sps.factorial(m) * sps.factorial(d)))

    # Define coefficient vector
    c = get_coefficient_vector(system, sys_params, m, p=p)

    # Find the true solution and derivative
    out = solve_ivp(run_monomial_ode, [0, t[-1]], u0, args=[c, d], t_eval=t,
                    rtol=1e-12, atol=1e-12)
    u_actual = out.y
    Theta = mlu.make_Theta(u_actual, d=d)
    udot_actual = (Theta @ c.T).T

    # Add noise
    np.random.seed(seed)
    u = u_actual + np.random.normal(0, np.sqrt(nu), (m, N))

    return u, u_actual, udot_actual, c


def get_system_values(system):
    """Get ODE parameters, initial conditions, and library degree."""
    if system == '1':
        sys_params = {}
        u0 = [0, 1]
        d = 3
    if system == '2a':
        sys_params = {'gamma': 0.1, 'kappa': 1, 'epsilon': 5}
        u0 = [0, 1]
        d = 4
    if system == '2b':
        sys_params = {'gamma': 0.2, 'kappa': 0.2, 'epsilon': 1}
        u0 = [0, 1]
        d = 4
    if system == '2c':
        sys_params = {'gamma': 0.2, 'kappa': 0.2, 'epsilon': 1}
        u0 = [0, 2]
        d = 4
    if system == '3':
        sys_params = {'gamma': 2}
        u0 = [0, 1]
        d = 4
    if system == '4':
        sys_params = {'alpha': .2, 'beta': .2, 'kappa': 5.7}
        u0 = [0, -5, 0]
        d = 2

    return (sys_params, u0, d)


def run_monomial_ode(t, u, c, d):
    """Run ODE system."""
    return (mlu.make_Theta(u, d=d) @ c.T).flatten()
