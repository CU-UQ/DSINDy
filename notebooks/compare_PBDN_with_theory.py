"""Compare numerical results with theoretical predicitons."""

# Set up
import os
import numpy as np
import numpy.linalg as la
import scipy.special as sps
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

import eqndiscov.denoising_functions as df
import eqndiscov.ODE_systems as sys
import eqndiscov.utils as utils
import eqndiscov.monomial_library_utils as mlu

# Pull out matplotlib colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rc('text.latex', preamble=r'\usepackage{bm}')

plt.rcParams['axes.autolimit_mode'] = "data"
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = .05

plt.rc('font', size=10)           # controls default text size
plt.rc('axes', titlesize=10)      # fontsize of the title
plt.rc('axes', labelsize=10)      # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)      # fontsize of the x tick labels
plt.rc('ytick', labelsize=8)      # fontsize of the y tick labels
plt.rc('legend', fontsize=5)      # fontsize of the legend

# %% USER DEFINED PARAMETERS
system = '4'
ttrain = 10
N_vec = np.logspace(2, 4, 5).astype(int)
# N_vec = np.logspace(2, 3, 3).astype(int)
alpha = .1
max_iter = 1000
out_dir = '/app/current_output/PSDN_Theory/'
nu_vec = [.1]

# %% Set up and plot initial states

# Make output directory if it doesn't exist already
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Obtain relevation parameters
sys_params, u0, d = sys.get_system_values(system)
out_file = f'PSDN_Theory_sys={system}_t={ttrain}_alp={alpha}.pdf'

m = np.size(u0)
P = int(sps.factorial(m + d) / (sps.factorial(m) * sps.factorial(d)))

use_actual_P = True
center_Theta = True
N = 1000
t = np.linspace(0, ttrain, num=N)
nu = 0.1
sigma = np.sqrt(nu)
u, u_actual, du_actual, c_actual = sys.setup_system(
    t, nu, d, system[0], sys_params=sys_params, u0=u0, seed=2)
for i in range(m):
    fig2, ax2 = plt.subplots(1)
    ax2.plot(t, u_actual[i, :])
    fig2.show()

# %%


def get_error_with_increasing_N(
        nu, N_vec, seed_vec, use_actual_P=False, alpha=1, max_iter=1,
        center_Theta=False, intervals=1, realizations=50):
    """Obtain smoothing error at multiple values of N."""
    u_err_mean_vec = []
    u_err_std_vec = []
    u_err_theory_vec = []
    u_err_mean_noise_vec = []
    u_err_std_noise_vec = []
    N_min = int(N_vec[0] / 10)
    N_max = int(N_vec[-1] * 10)

    # Add theoretical values at N_Min\
    t = np.linspace(0, ttrain, num=N_min)
    # Find noisy/actual measurements and actual derivative/coef vector
    sigma = np.sqrt(nu)
    u, u_actual, du_actual, c_actual = sys.setup_system(
        t, nu, d, system[0], sys_params=sys_params, u0=u0, seed=0)
    u_err_theory_vec.append(
        np.sqrt(nu / la.norm(u_actual, axis=1)**2 * (P + 1)))
    # u_err_mean_noise_vec.append(utils.rel_err(u, u_actual))
    u_err_mean_noise_vec.append(
        np.sqrt(nu * N_min) * 1 / la.norm(u_actual, axis=1))
    for i, N in enumerate(N_vec):
        u_err_vec = []
        u_err_noise_vec = []
        for j in range(realizations):
            seed = seed_vec[i]
            t = np.linspace(0, ttrain, num=N)
            # Find noisy/actual measurements and actual derivative/coef vector
            sigma = np.sqrt(nu)
            u, u_actual, du_actual, c_actual = sys.setup_system(
                t, nu, d, system[0], sys_params=sys_params, u0=u0,
                seed=seed + j)

            # Projection-based smoothing
            A = utils.get_discrete_integral_matrix(t)
            u_proj = df.projection_denoising(
                u, u_actual, d, sigma * np.ones(m), A, alpha=alpha,
                max_iter=max_iter, plot=True, use_actual_P=use_actual_P,
                center_Theta=center_Theta)[0]

            # Save relative error
            u_err = utils.rel_err(np.array(u_proj), u_actual)
            u_err_vec.append(u_err)

            # Error of original noisy data
            u_err_noise_vec.append(utils.rel_err(u, u_actual))

        # Save mean/std relative error
        u_err_mean_vec.append(np.mean(np.array(u_err_vec), axis=0))
        u_err_std_vec.append(np.std(np.array(u_err_vec), axis=0))
        u_err_std_noise_vec.append(np.std(np.array(u_err_noise_vec), axis=0))
        u_err_mean_noise_vec.append(
            np.sqrt(nu * N) * 1 / la.norm(u_actual, axis=1))

        # Theoretical error expectation (does not include quadrature error)
        u_err_theory_vec.append(
            np.sqrt(nu / la.norm(u_actual, axis=1)**2 * (P + 1)))
    # Add theoretical values at N_max
    t = np.linspace(0, ttrain, num=N_max)
    # Find noisy/actual measurements and actual derivative/coef vector
    sigma = np.sqrt(nu)
    u, u_actual, du_actual, c_actual = sys.setup_system(
        t, nu, d, system[0], sys_params=sys_params, u0=u0, seed=0)
    u_err_theory_vec.append(
        np.sqrt(nu / la.norm(u_actual, axis=1)**2 * (P + 1)))
    # u_err_mean_noise_vec.append(utils.rel_err(u, u_actual))
    u_err_mean_noise_vec.append(
        np.sqrt(nu * N_max) * 1 / la.norm(u_actual, axis=1))
    u_err_mean_vec = np.array(u_err_mean_vec).T
    u_err_std_vec = np.array(u_err_std_vec).T

    u_err_theory_vec = np.array(u_err_theory_vec).T
    u_err_mean_noise_vec = np.array(u_err_mean_noise_vec).T
    u_err_std_noise_vec = np.array(u_err_std_noise_vec).T

    return (
        u_err_mean_vec, u_err_std_vec, u_err_mean_noise_vec,
        u_err_std_noise_vec, u_err_theory_vec)


# %% PLOT SHOWING CONVERGENCE WHEN WE KNOW TRUE PROJECTION


def plot_results(u_errors, col, sig):
    """Plot the three error types."""
    N_vec_large = np.concatenate(([N_vec[0] / 10], N_vec, [N_vec[-1] * 10]))
    for i, ui in enumerate(u_errors[0]):
        print(i)
        print(ui)
        print(u_errors[1][i])
        axs[i, col].errorbar(
            N_vec, ui, yerr=u_errors[1][i], fmt='.',
            label=r'$\mathcal{E}(\tilde{\bm{u}}$'
            rf'$_{i+1};$'
            r'$\bm{u}$'
            fr'$_{i+1}^*)$', color=colors[i])
        axs[i, col].plot(
            N_vec_large, u_errors[2][i], label=r'$\mathcal{E}_{noisy}(\bm{u}$'
            fr'$_{i+1}^*$)', color=colors[i], linestyle='dashed')
        axs[i, col].plot(
            N_vec_large, u_errors[4][i], label=r'$\mathcal{E}_{theory}(\bm{u}$'
            fr'$_{i+1}^*$)', color=colors[i], linestyle='dotted')


# %% Obtain smoothing results for the 10 simulations

u_errors_a = {}
u_errors_b = {}
u_errors_c = {}
seed_vec = [
    991992, 102488, 793759, 807345, 948548, 802870, 818898, 909001, 491821
]
for j, nu in enumerate(nu_vec):
    sig = np.sqrt(nu)
    u_errors_a[j] = get_error_with_increasing_N(
        nu, N_vec, seed_vec, use_actual_P=True)
    u_errors_b[j] = get_error_with_increasing_N(
        nu, N_vec, seed_vec, use_actual_P=False, max_iter=1,
        center_Theta=center_Theta)
    u_errors_c[j] = get_error_with_increasing_N(
        nu, N_vec, seed_vec, use_actual_P=False, alpha=alpha,
        max_iter=max_iter, center_Theta=center_Theta)

# %%
fig, axs = plt.subplots(m, 3)
if system == '4':
    fig.set_size_inches(5, .5 + 1.5 * m)
else:
    fig.set_size_inches(5, .5 + 1.6 * m)

for j in range(np.size(nu_vec)):
    plot_results(u_errors_a[j], 0, sig)
    plot_results(u_errors_b[j], 1, sig)
    plot_results(u_errors_c[j], 2, sig)

# Set up axes
for j in range(3):
    for i in range(m):
        axs[i, j].loglog()
        axs[i, j].grid(True, which='major', color='gray')
        axs[i, j].set_xlim([10**1.8, 10**4.2])

        # axs[i, j].set_ylim([0.0002, 0.2])

# Set up axes labels
for i in range(m):
    axs[i, 0].set_ylabel(r'Relative $\tilde{\bm{u}}$' + fr'$_{i + 1}$ error')

for i in range(3):
    axs[m - 1, i].set_xlabel(r'$N$')

# Set up shared x-axes
for j in range(3):
    for i in range(0, m - 1):
        axs[i, j].get_shared_x_axes().join(axs[m - 1, j], axs[i, j])
        axs[i, j].xaxis.set_ticklabels([])

# Set up shared y-axes
for j in range(1, 3):
    for i in range(m):
        axs[i, j].get_shared_y_axes().join(axs[i, 0], axs[i, j])
        axs[i, j].yaxis.set_ticklabels([])

if system == '4':
    axs[1, 0].set_ylim([4 * 10**-4, .08])
    axs[2, 0].set_ylim([10**-3, .2])

# axs[0, 2].set_ylim([10**-4, 10**-1])

axs[0, 0].set_title(r'$\tilde{\bm{u}}=P_{\Phi^*}\bm{u}$')
axs[0, 1].set_title(r'$\tilde{\bm{u}}=P_{\Phi}\bm{u}$')
axs[0, 2].set_title(r'$\tilde{\bm{u}}={\tt IterPSDN}(\alpha=$'
                    f'{alpha})')
# for j in range(3):
#     axs[0, j].title.set_size(10)

for i in range(m):
    axs[i, 0].legend()

# Lines, Labels = axs[0, 0].get_legend_handles_labels()
# idx = np.argsort(Labels)
# LabelsNew = [Labels[i] for i in list(idx)]
# LinesNew = [Lines[i] for i in list(idx)]
#
# fig.legend(LinesNew, LabelsNew, loc='upper center',
#            bbox_to_anchor=(0.5, 0), ncol=3, handlelength=4)

fig.set_dpi(600.0)
fig.tight_layout(rect=[0.01, 0.01, 0.01, 0.01])
plt.savefig(f'{out_dir}/{out_file}', pad_inches=0)
fig.show()

# %%
with open(f'{out_dir}/{out_file[:-4]}_a.pkl', 'wb') as fh:
    pickle.dump(u_errors_a, fh)
with open(f'{out_dir}/{out_file[:-4]}_b.pkl', 'wb') as fh:
    pickle.dump(u_errors_b, fh)
with open(f'{out_dir}/{out_file[:-4]}_c.pkl', 'wb') as fh:
    pickle.dump(u_errors_c, fh)

# %% Convergange of Phi to Phi_star

system = '4'
ttrain = 10
N = 1000
sys_params, u0, d = sys.get_system_values(system)
nu = 0.01
ii = 0
Psi_norm = []
Psi_dagger_norm = []
Delta_Psi_norm = []
N_vec2 = np.logspace(1, 4, 5).astype(int)
for N in N_vec2:
    N = N + 1
    t = np.linspace(0, ttrain, num=N)
    A = utils.get_discrete_integral_matrix(t)
    u, u_actual, du_actual, c_actual = sys.setup_system(
        t, nu, d, system[0], sys_params=sys_params, u0=u0, seed=1)

    Theta_actual = mlu.make_Theta(u_actual, d=d)
    Theta = mlu.make_Theta(u, d=d)

    Phi = A @ Theta
    bmone = np.ones(N).reshape(-1, 1)
    Psi = np.hstack((bmone, Phi)) / np.sqrt(N)
    Phi_actual = A @ Theta_actual
    Psi_actual = np.hstack((bmone, Phi_actual)) / np.sqrt(N)

    Psi_norm.append(la.norm(Psi_actual))
    Psi_dagger_norm.append(la.norm(la.pinv(Psi_actual)))
    Delta_Psi_norm.append(la.norm(Psi - Psi_actual))

# Plot results
fig2, ax2 = plt.subplots(3, 1)
ax2[0].plot(N_vec2, Psi_norm, label=r'\|\Psi^*\|')
ax2[1].plot(N_vec2, Psi_dagger_norm, label=r'\|(\Psi^*)^\dagger\|')
ax2[2].plot(N_vec2, Delta_Psi_norm, label=r'\|\Delta \Psi\|')
for i in range(3):
    ax2[i].set_ylabel('Norm value')
    ax2[i].set_xlabel('N')
    ax2[i].loglog()
fig2.show()

# %% Variance of library and bound

system = '4'
ttrain = 15
sys_params, u0, d = sys.get_system_values(system)
m = np.size(u0)
nu = 0.01
N = 1000
n_rep = 100
t = np.linspace(0, ttrain, num=N)
seed = 998711

# Find a variance estimate using mulitple Theta realizations
Thetas = []
for i in range(n_rep):
    u, u_actual, du_actual, c_actual = sys.setup_system(
        t, nu, d, system[0], sys_params=sys_params, u0=u0, seed=seed)
    Theta_temp = mlu.make_Theta(u, d=d)
    Theta = mlu.center_Theta(Theta_temp, d, m, nu)
    Thetas.append(Theta)
Theta_actual = mlu.make_Theta(u_actual, d=d)
Theta_var = np.zeros(Theta_actual.shape)
for td in Thetas:
    Theta_var += (td - Theta_actual)**2
Theta_var = 1 / n_rep * Theta_var
np.max(Theta_var, axis=0)

# Find variance bound for each basis element
mi_mat = mlu.make_mi_mat(m, d)
p = np.size(mi_mat, 0)
var_bounds = []
for mi in mi_mat[1:]:
    prod1 = 1
    prod2 = 1
    for k in range(m):
        sum1 = 0
        sum2 = 0
        for j in range(0, mi[k] + 1):
            if j <= mi[k] / 2:
                u_term1 = u_actual[k, :]**(mi[k] - 2 * j)
                c1 = utils.ncr(mi[k], 2 * j)
                sum1 += u_term1 * c1 * nu**j * np.prod(range(2 * j - 1, 0, -2))
            u_term2 = u_actual[k, :]**(2 * mi[k] - 2 * j)
            c2 = utils.ncr(2 * mi[k], 2 * j)
            sum2 += u_term2 * c2 * nu**j * np.prod(range(2 * j - 1, 0, -2))

        prod1 *= sum1
        prod2 *= sum2

    var_bounds.append(np.max(prod2 - prod1**2))

fig3, ax3 = plt.subplots()
ax3.plt(range(1, p), np.max(Theta_var, axis=0)[1:])
ax3.plt(range(1, p), np.array(var_bounds))
fig3.show()
