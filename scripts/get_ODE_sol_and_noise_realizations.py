"""Generate noise realizations and smoothed results for given ODE systems.

Possible ODE systems include:
    (1) Simple Harmonic Oscillator
    (2) Duffing Oscillator
    (3) Van der Pol Oscillator
    (4) Rossler Attractor
    (5) Lorenz 96

The actual, noisy, Gaussian Process smoothed, and projection-based smoothed
results are saved to .csv file.

"""
# %%
import os
import random
import numpy as np
import numpy.linalg as la
import pandas as pd

import dsindy.denoising_functions as df
import dsindy.ODE_systems as odesys
import dsindy.utils as utils
# import plotly.graph_objects as go


def run_replications(system,
                     N,
                     ttrain,
                     start=0,
                     add_to_file=False,
                     find_gp=True,
                     replications=20,
                     bdir='/app/'):
    """Smoothing noise replications."""
    # Number of sample replications
    max_iter = 1000
    alpha = 0.1
    outdir = f'{bdir}/current_output/smoothed_noise_realizations/'
    if system[0] == '2':
        indir = f'{bdir}/paper_noise_realizations/Duffing/'
    if system[0] == '3':
        indir = f'{bdir}/paper_noise_realizations/Van_der_Pol/'
    if system[0] == '4':
        indir = f'{bdir}/paper_noise_realizations/Rossler/'
    if system[0] == '5':
        indir = f'{bdir}/paper_noise_realizations/Lorenz96/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Find the noise level/random seed vectors
    # nu_vec = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    nu_vec = [10]

    if system == '2a' and N == 1000 and ttrain == 10:
        rand_seed = 991182
    if system == '2b' and N == 250 and ttrain == 10:
        rand_seed = 688871
    if system == '2b' and N == 500 and ttrain == 10:
        rand_seed = 299823
    if system == '2b' and N == 1000 and ttrain == 10:
        rand_seed = 280011
    if system == '2b' and N == 2000 and ttrain == 10:
        rand_seed = 221011
    if system == '2b' and N == 4000 and ttrain == 10:
        rand_seed = 209909
    if system == '2b' and N == 8000 and ttrain == 10:
        rand_seed = 219029
    if system == '2c' and N == 1000 and ttrain == 30:
        rand_seed = 280011

    if system == '3' and N == 1000 and ttrain == 10:
        rand_seed = 366765

    if system == '4' and N == 1000 and ttrain == 10:
        rand_seed = 464570
    if system == '4' and N == 2000 and ttrain == 10:
        rand_seed = 403611

    if system == '5' and N == 1000 and ttrain == 5:
        rand_seed = 102921
    if system == '5' and N == 2000 and ttrain == 5:
        rand_seed = 102922

    # Set random seeds for system
    random.seed(rand_seed)
    seed_vec = [random.randint(0, 1e6) for i in range(len(nu_vec))]

    # Set up system
    sys_params, u0, p = odesys.get_system_values(system)

    # Iterate through noise values, find noise realization, perform smoothing
    for run_noise_level in range(np.size(nu_vec)):

        # Pull out varaince and random seed
        nu = nu_vec[run_noise_level]
        seed = seed_vec[run_noise_level]

        infn = f'{indir}/system={system}_ttrain={ttrain}_N={N}_nu={nu}.csv'
        outfn = f'{outdir}/system={system}_ttrain={ttrain}_N={N}_nu={nu}.csv'

        # Define parameters
        m = np.size(u0)

        # Time vector when measurements are taken
        t = np.linspace(0, ttrain, num=N)

        # Initialize dataframe which we will contain the results
        if add_to_file:
            data_all = pd.read_csv(infn)
        else:
            data_all = pd.DataFrame({'time': t})

        for j in range(start, start + replications):
            # Find noisy/actual measurements and actual derivative/coef vector
            u, u_actual, du_actual, c_actual = odesys.setup_system(
                t,
                nu,
                p,
                system[0],
                sys_params=sys_params,
                u0=u0,
                seed=seed + j)

            # Gaussian process smooth data
            if find_gp:
                u_smooth = df.smooth_data(t, u)
                sigma_estimate = 1 / np.sqrt(N) * la.norm(u_smooth - u, axis=1)
                check_diverge = True
                print(np.abs(sigma_estimate - np.sqrt(nu)) / np.sqrt(nu))
            else:
                # Find GP using only 2000 datapoints (i.e., subsample)
                # sub_samp_rate = int(N / 2000)
                # u_smooth = df.smooth_data(
                #     t[::sub_samp_rate], u[:, ::sub_samp_rate])
                # N_samp = np.size(u_smooth[0])
                # sigma_estimate = 1 / np.sqrt(N_samp) * la.norm(
                #     u_smooth - u[:, ::sub_samp_rate], axis=1)
                #
                # print(np.abs(sigma_estimate - np.sqrt(nu)) / np.sqrt(nu))
                check_diverge = False
                sigma_estimate = np.zeros(m)
                # sigma_estimate = np.sqrt(nu) * np.ones(m)

            # Projection-based smoothing
            u_proj = np.empty((m, N))

            # Projection-based smoothing
            A = utils.get_discrete_integral_matrix(t)
            u_proj = df.projection_denoising(u,
                                             u_actual,
                                             p,
                                             sigma_estimate,
                                             A,
                                             alpha=alpha,
                                             max_iter=max_iter,
                                             plot=True,
                                             use_actual_P=False,
                                             center_Theta=True,
                                             check_diverge=check_diverge)[0]

            # fig = go.Figure()
            # fig.add_trace(go.Scatter(x=t, y=u_proj[0]))
            # fig.add_trace(go.Scatter(x=t, y=u_proj[1]))
            # fig.show()

            # At first iteration add the actual resutlts
            if j == 0:
                for i in range(m):
                    data_all[f'actual-u{i+1}'] = u_actual[i]

            # Add the noisy and smoothed results
            for i in range(m):
                data_all[f'S{j+1}-noise-u{i+1}'] = u[i]
            if find_gp:
                for i in range(m):
                    data_all[f'S{j+1}-gp-u{i+1}'] = u_smooth[i]
            for i in range(m):
                data_all[f'S{j+1}-proj-u{i+1}'] = u_proj[i]

        # Save results to csv file
        # outfn = f'{outdir}/system={system}_ttrain={ttrain}_N={N}_nu={nu}.csv'
        data_all.to_csv(outfn, index=False)


# %% Generate smoothed data for each system

# For running additional simulations
start = 0
add_to_file = False
bdir = '/app/'

system = '2a'
N = 1000  # Number of samples
ttrain = 10  # Training time

run_replications(system,
                 N,
                 ttrain,
                 start=start,
                 add_to_file=add_to_file,
                 replications=10,
                 bdir=bdir)

system = '2b'
N = 250  # Number of samples
ttrain = 10  # Training time

run_replications(system,
                 N,
                 ttrain,
                 start=start,
                 add_to_file=add_to_file,
                 bdir=bdir)

system = '2b'
N = 500  # Number of samples
ttrain = 10  # Training time

run_replications(system,
                 N,
                 ttrain,
                 start=start,
                 add_to_file=add_to_file,
                 bdir=bdir)

system = '2b'
N = 1000  # Number of samples
ttrain = 10  # Training time

run_replications(system,
                 N,
                 ttrain,
                 start=start,
                 add_to_file=add_to_file,
                 bdir=bdir)

system = '2b'
N = 2000  # Number of samples
ttrain = 10  # Training time

run_replications(system,
                 N,
                 ttrain,
                 start=start,
                 add_to_file=add_to_file,
                 bdir=bdir)

system = '2b'
N = 4000  # Number of samples
ttrain = 10  # Training time

run_replications(system,
                 N,
                 ttrain,
                 start=start,
                 add_to_file=add_to_file,
                 bdir=bdir)

# For this sample size don't perform Gaussian process regression, too slow
system = '2b'
N = 8000  # Number of samples
ttrain = 10  # Training time

run_replications(system,
                 N,
                 ttrain,
                 start=0,
                 find_gp=False,
                 replications=30,
                 bdir=bdir)

system = '3'
N = 1000  # Number of samples
ttrain = 10  # Training time

run_replications(system,
                 N,
                 ttrain,
                 start=start,
                 add_to_file=add_to_file,
                 bdir=bdir)

system = '4'
N = 1000  # Number of samples
ttrain = 10  # Training time

run_replications(system,
                 N,
                 ttrain,
                 start=start,
                 add_to_file=add_to_file,
                 bdir=bdir)

# %%
system = '5'
N = 2000  # Number of samples
ttrain = 5  # Training time
start = 0
add_to_file = False
bdir = '/home/jacqui/projects/DSINDy/'

run_replications(system,
                 N,
                 ttrain,
                 start=start,
                 add_to_file=add_to_file,
                 replications=2,
                 bdir=bdir)

# %%
