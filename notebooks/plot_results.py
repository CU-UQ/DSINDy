# ---
# jupyter:
#     authors:
#         [name: Jremoteacqui Wentz]
#     title: "Duffing oscillator plots"
#     jupytext:
#         cell_markers:
#
# ---
"""This notebook plots the summary results."""

# %% [markdown]
"""
# Results summary.

In this notebook I plot summary results.

I look at four different noise levels and at each noise level learned the
system using 30 realizations of the noise.
The plots show the mean results of these 10 realizations.

"""


# %%
import os
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import (MultipleLocator, LogLocator, PercentFormatter)

mpl.rc('text.latex', preamble=r'\usepackage{bm}')
plt.rcParams['axes.autolimit_mode'] = "data"
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.rc('font', size=10)  # controls default text size
plt.rc('axes', titlesize=10)  # fontsize of the title
plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)  # fontsize of the x tick labels
plt.rc('ytick', labelsize=8)  # fontsize of the y tick labels
plt.rc('legend', fontsize=5)  # fontsize of the legend

# %%

# system = '2a'
system = '2b'
# system = '3'
# system = '4'

# %%


def generate_df(nu_vec, N=1000, ttrain=10, system='2', reps=30):
    """Iterate through simulation realizations and extract mean/std."""
    if system[0] == '2':
        base_dir = '/app/paper_optimizations/Duffing/'
    if system[0] == '3':
        base_dir = '/app/paper_optimizations/Van_der_Pol/'
    if system[0] == '4':
        base_dir = '/app/paper_optimizations/Rossler/'
    firstRow = False
    lasso_failed = []
    socp_failed = []
    socp_theory_failed = []
    for k in range(len(nu_vec)):
        nu = nu_vec[k]
        base_fn = f'system={system}_nu={nu}_N={N}_ttrain={ttrain}'
        firstReal = False
        for i in range(reps):
            fn = f'{base_dir}/{base_fn}/{base_fn}_realization={i}.csv'
            if firstReal is False:
                try:
                    err_df = pd.read_csv(
                        fn, header=None, index_col=[0], names=[i])
                    firstReal = True
                except Exception:
                    print(f'Results for nu={nu} realization={i} do not exist.')
                    continue
            else:
                try:
                    err_df_i = pd.read_csv(fn, header=None, index_col=[0],
                                           names=[i])
                    err_df = err_df.merge(err_df_i, right_index=True,
                                          left_index=True)
                # Print iteration if it doesn't exist (means it failed)
                except Exception:
                    print(f'Results for nu={nu} realization={i} do not exist.')
                    continue

        n = np.size(err_df, 1)

        err_df[err_df == -1] = 1  # None
        err_df[err_df > 1] = 1  # None

        key_ends = ['_pred_socp_sm',
                    '_pred_socp_theory',
                    '_pred_lasso']
        for key_end in key_ends:
            keyset = []
            for k in range(m):
                keyset = keyset + [f'u{k+1}{key_end}']
            failed = np.mean(np.max(err_df.loc[keyset], axis=0) == 1)
            if key_end == '_pred_socp_sm':
                socp_failed.append(failed)
            if key_end == '_pred_socp_theory':
                socp_theory_failed.append(failed)
            if key_end == '_pred_lasso':
                lasso_failed.append(failed)

        print(f'nu={nu}')
        print(n)

        if firstRow is False:
            df_mean = pd.DataFrame(err_df.mean(axis=1)).T
            df_std = pd.DataFrame(err_df.std(axis=1)).T
            df_sem = pd.DataFrame(err_df.sem(axis=1)).T
            firstRow = True
        else:
            df_mean_k = pd.DataFrame(err_df.mean(axis=1)).T
            df_std_k = pd.DataFrame(err_df.std(axis=1)).T
            df_sem_k = pd.DataFrame(err_df.sem(axis=1)).T
            df_mean_k.index = [k]
            df_mean = pd.concat([df_mean, df_mean_k])
            df_std_k.index = [k]
            df_std = pd.concat([df_std, df_std_k])
            df_sem_k.index = [k]
            df_sem = pd.concat([df_sem, df_sem_k])

    # Set the varaince as the row name
    df_mean.index = np.sqrt(nu_vec)
    df_std.index = np.sqrt(nu_vec)
    df_sem.index = np.sqrt(nu_vec)

    df_mean['lasso_failed'] = lasso_failed
    df_mean['socp_failed'] = socp_failed
    df_mean['socp_theory_failed'] = socp_theory_failed
    return(df_mean, df_std, df_sem)


def gen_plot(cols, dfs_mean, dfs_std, ylabel, N_vec, labels=None, fn='temp',
             colors=None, swap=False, logy=True, include_std=True, ypad=4):
    """Function to generate plots of data-driven dynamics results."""
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linetype = ['solid', 'dashed', 'dotted']

    """Plot the results as a function of noise level."""
    fig, axs = plt.subplots(1, len(dfs_mean), sharey=True)
    if len(dfs_mean) > 1:
        fig.set_size_inches(2 * len(dfs_mean), 2)
    else:
        fig.set_size_inches(1.75, 2)
    if type(axs) is not np.ndarray:
        axs = [axs]
    # Reorganize datafame to more easily plot
    dfs_sig = {}
    for i in range(len(dfs_mean)):
        df_new = dfs_mean[i][cols]
        df1 = pd.melt(df_new.reset_index(), id_vars='index')
        df1.rename(columns={'index': 'sigma', 'value': 'mean'}, inplace=True)
        if include_std:
            df_std_new = dfs_std[i][cols]
            df2 = pd.melt(df_std_new.reset_index(), id_vars='index')
            df2.rename(columns={'index': 'sigma', 'value': 'std'},
                       inplace=True)
            df1['std'] = df2['std']

        if i == 0:
            sig_vec = [0.01, 0.1, 1]
        for j, key in enumerate(cols):
            if include_std:
                df1[df1['variable'] == key].plot(
                    'sigma', 'mean', yerr='std', label=labels[key], ax=axs[i],
                    marker='.', color=colors[j], linestyle=linetype[j])
            else:
                df1[df1['variable'] == key].plot(
                    'sigma', 'mean', label=labels[key], ax=axs[i],
                    marker='.', color=colors[j], linestyle=linetype[j])

        if logy:
            axs[i].loglog()
            axs[i].set_title(f'N={N_vec[i]}')
        else:
            axs[i].semilogx()

        axs[i].set_xlabel(r'$\sigma$')
        axs[i].grid(True, which='major', color='gray')
        # axs[i].set_xticklabels([0.001, 0.01, .1, 1])
        axs[i].set_xticks([0.001, 0.01, .1, 1])
        axs[i].get_legend().remove()
        axs[i].xaxis.set_minor_locator(LogLocator(
            base=10, subs=np.arange(2, 10) * .1, numticks=100))

        if not logy:
            axs[i].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            axs[i].yaxis.set_minor_locator(MultipleLocator(.2))
            axs[i].yaxis.set_minor_locator(MultipleLocator(.1))
            axs[i].set_ylim([-0.05, 1.05])
            if i == 0:
                axs[i].set_title(ylabel)

        df1['N'] = N_vec[i]
        for j, sig in enumerate(np.unique(df1.sigma)):
            if i == 0:
                dfs_sig[j] = df1[df1.sigma == sig]
            else:
                dfs_sig[j] = pd.concat([dfs_sig[j], df1[df1.sigma == sig]])
            dfs_sig[j].pop('sigma')
    fig.set_dpi(600.0)
    plt.savefig('/app/current_output/plots/' + fn + '.pdf',
                pad_inches=0)
    plt.show()

    """Plot the results as a function of noise level."""
    if len(dfs_mean) > 1:
        fig, axs = plt.subplots(1, len(dfs_sig), sharey=True)
        fig.set_size_inches(1.75 * len(dfs_sig), 2)
        if type(axs) is not np.ndarray:
            axs = [axs]
        # Reorganize datafame to more easily plot
        for i in range(len(dfs_sig)):
            df1 = dfs_sig[i]
            for j, key in enumerate(cols):
                if logy:
                    df1[df1['variable'] == key].plot(
                        'N', 'mean', yerr='std', label=labels[key], ax=axs[i],
                        marker='.', color=colors[j], linestyle=linetype[j])
                else:
                    df1[df1['variable'] == key].plot(
                        'N', 'mean', label=labels[key], ax=axs[i],
                        marker='.', color=colors[j], linestyle=linetype[j])
            if logy:
                axs[i].loglog()
            else:
                axs[i].semilogx()

            if i == 0:
                axs[i].set_ylabel(ylabel, labelpad=ypad)

            axs[i].set_xlabel('N')
            axs[i].grid(True, which='major', color='gray')
            axs[i].set_xticks([1000])
            axs[i].set_xticks([250, 500, 2000, 4000, 8000], minor=True)
            axs[i].set_xticklabels([1000], rotation=45)
            axs[i].set_xticklabels([250, 500, 2000, 4000, 8000], minor=True,
                                   rotation=45)
            for j in range(5):
                axs[i].xaxis.get_minorticklabels()[j].set_y(-.02)

            if not logy:
                axs[i].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
                axs[i].yaxis.set_minor_locator(MultipleLocator(.2))
                axs[i].yaxis.set_minor_locator(MultipleLocator(.1))
                axs[i].set_ylim([-0.05, 1.05])
            else:
                axs[i].yaxis.set_major_locator(LogLocator(
                    base=10, numticks=100))
                axs[i].yaxis.set_minor_locator(LogLocator(
                    base=10, subs=np.arange(2, 10) * .1, numticks=100))

            axs[i].set_title(fr'$\sigma$={sig_vec[i]:.2f}')
            axs[i].get_legend().remove()

        fig.set_dpi(600.0)
        plt.savefig('/app/current_output/plots/' + fn + '_swap.pdf',
                    pad_inches=0)
        plt.show()


def gen_plot_set(cols_temp, df_mean, df_std, labels=None,
                 fn='temp', error_type='c', ylabel_start='',
                 colors=None, swap=False, logy=True, include_std=True, ypad=4):
    """Function to generate plots of data-driven dynamics results."""
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    linetype = ['solid', 'dashed', 'dotted']
    """Plot the results as a function of noise level."""
    if error_type == 'du' or error_type == 'us':
        fig, axs = plt.subplots(1, 3, sharey=True)
        for i in range(np.size(axs)):
            axs[i].set_xlim(10**(-3.2), 10**(0.2))
        fig.set_size_inches(5.25, 2)
    else:
        fig, axs = plt.subplots(1, m, sharey=True)
        fig.set_size_inches(1.75 * m, 2)

    # Reorganize datafame to more easily plot
    for i in range(m):
        if error_type == 'c':
            cols = [f'c{i + 1}_' + lab for lab in cols_temp]
        if error_type == 'u' or error_type == 'us':
            cols = [f'u{i + 1}_' + lab for lab in cols_temp]
        if error_type == 'du':
            cols = [f'du{i + 1}_' + lab for lab in cols_temp]
        df_new = df_mean[cols]
        df1 = pd.melt(df_new.reset_index(), id_vars='index')
        df1.rename(columns={'index': 'sigma', 'value': 'mean'}, inplace=True)
        df_std_new = df_std[cols]
        df2 = pd.melt(df_std_new.reset_index(), id_vars='index')
        df2.rename(columns={'index': 'sigma', 'value': 'std'},
                   inplace=True)
        df1['std'] = df2['std']
        for j, key in enumerate(cols):
            key_old = cols_temp[j]
            df1[df1['variable'] == key].plot(
                'sigma', 'mean', yerr='std', label=labels[key_old], ax=axs[i],
                marker='.', color=colors[j], linestyle=linetype[j])
        axs[i].get_legend().remove()

    if error_type == 'c':
        ylabel_add = r'Relative $\bm{c}_k$ error'
    if error_type == 'u':
        ylabel_add = r'Relative $\bm{u}_k$ error'
    if error_type == 'du':
        ylabel_add = r'Relative $\dot{\bm{u}}_k$ error'
    if error_type == 'us':
        ylabel_add = r'Relative $\tilde{\bm{u}}_k$ error'
    axs[0].set_ylabel(f'{ylabel_start}{ylabel_add}')

    for i in range(np.size(axs)):
        axs[i].loglog()
        axs[i].set_xlabel(r'$\sigma$')
        axs[i].grid(True, which='major', color='gray')
        axs[i].set_xticks([0.001, 0.01, .1, 1])
        axs[i].yaxis.set_major_locator(LogLocator(base=10, numticks=100))
        axs[i].yaxis.set_minor_locator(LogLocator(
            base=10, subs=np.arange(2, 10) * .1, numticks=100))
        axs[i].xaxis.set_minor_locator(LogLocator(
            base=10, subs=np.arange(2, 10) * .1, numticks=100))
        axs[i].set_title(rf'$k={i + 1}$')
        if i > m - 1:
            # hide axis
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            for spine in ['top', 'right', 'left', 'bottom']:
                axs[i].spines[spine].set_visible(False)

    # Save legend as separate image
    label_params = axs[0].get_legend_handles_labels()
    fig.set_dpi(600.0)
    plt.savefig('/app/current_output/plots/' + fn + '.pdf', pad_inches=0)
    plt.show()

    # Make legend figures
    collist = '_'.join(cols_temp)
    figl, axl = plt.subplots()
    axl.axis(False)
    legend = axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5),
                        ncol=1, handlelength=4, fontsize=8)  # len(cols))
    legend_fig = legend.figure
    legend_fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-3, -3, 3, 3])))
    bbox = bbox.transformed(legend_fig.dpi_scale_trans.inverted())
    legend_fig.savefig(f'/app/current_output/plots/legends/legend_{collist}.pdf',
                       bbox_inches=bbox)

    # Make legend figures (horizontal)
    collist = '_'.join(cols_temp)
    figl, axl = plt.subplots()
    axl.axis(False)
    legend = axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5),
                        ncol=len(collist), handlelength=4, fontsize=8)
    legend_fig = legend.figure
    legend_fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-3, -3, 3, 3])))
    bbox = bbox.transformed(legend_fig.dpi_scale_trans.inverted())
    legend_fig.savefig(f'/app/current_output/plots/legends/legend_{collist}_h.pdf',
                       bbox_inches=bbox)


# %%
if system == '2a':
    nu_vec = [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1]
    m = 2
    compare_with_wsindy = True
    ttrain = 10
    N_vec = [1000]
    wsindy_dir = '/app/wsindy_results/Duffing'
    SysName = r'\textbf{Duffing (PS1)}'
if system == '2b':
    nu_vec = [0.0001, 0.01, 1]
    m = 2
    compare_with_wsindy = True
    ttrain = 10
    N_vec = [250, 500, 1000, 2000, 4000, 8000]
    wsindy_dir = '/app/wsindy_results/Duffing'
    SysName = r'\textbf{Duffing (PS2)}'
if system == '2c':
    nu_vec = [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1]
    system = '2c'
    m = 2
    compare_with_wsindy = True
    ttrain = 30
    N_vec = [1000]
    wsindy_dir = '/app/wsindy_results/Duffing'
if system == '3':
    nu_vec = [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1]
    m = 2
    compare_with_wsindy = True
    wsindy_dir = '/app/wsindy_results/Van_der_Pol'
    ttrain = 10
    N_vec = [1000]
    SysName = r'\textbf{Van der Pol}'
if system == '4':
    nu_vec = [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1]
    m = 3
    compare_with_wsindy = True
    wsindy_dir = '/app/wsindy_results/Rossler'
    ttrain = 10
    N_vec = [1000]
    SysName = r'\textbf{R\"{o}ssler}'



# %% Parse through results and save to dataframe

reps = 30
base_dir = f'system={system}'
if not os.path.exists(f'/app/current_output/plots/{base_dir}'):
    os.makedirs(f'/app/current_output/plots/{base_dir}')
if not os.path.exists(f'/app/current_output/plots/legends/'):
    os.makedirs(f'/app/current_output/plots/legends/')


dfs_mean = []
dfs_std = []
dfs_sem = []
for N in N_vec:
    df_mean, df_std, df_sem = generate_df(nu_vec, system=system, N=N,
                                  ttrain=ttrain)
    dfs_mean.append(df_mean)
    dfs_std.append(df_std)
    dfs_sem.append(df_sem)

# Add columns for WSINDY. ttrain=10
if compare_with_wsindy:

    for i, N in enumerate(N_vec):

        mean_wsindy = pd.read_csv(
            f'{wsindy_dir}/{base_dir}_ttrain=10_N={N}_mean_coef_err.csv',
            header=None, index_col=0)
        std_wsindy = pd.read_csv(
            f'{wsindy_dir}/{base_dir}_ttrain=10_N={N}_std_coef_err.csv',
            header=None, index_col=0)

        mean_u_wsindy = pd.read_csv(
            f'{wsindy_dir}/{base_dir}_ttrain=10_N={N}_mean_u_err.csv',
            header=None, index_col=0)
        std_u_wsindy = pd.read_csv(
            f'{wsindy_dir}/{base_dir}_ttrain=10_N={N}_std_u_err.csv',
            header=None, index_col=0)

        n_complete = mean_u_wsindy.loc[nu_vec][m + 1].to_numpy()
        dfs_mean[i]['wsindy_failed'] = n_complete / reps
        for j in range(1, m + 1):
            meani = mean_wsindy.loc[nu_vec][j].to_numpy()
            stdi = std_wsindy.loc[nu_vec][j].to_numpy()
            semi = std_wsindy.loc[nu_vec][j].to_numpy() / np.sqrt(reps)
            dfs_mean[i][f'c{j}_WSINDY'] = meani
            dfs_std[i][f'c{j}_WSINDY'] = stdi
            dfs_sem[i][f'c{j}_WSINDY'] = semi
            meanui = mean_u_wsindy.loc[nu_vec][j].to_numpy()
            stdui = std_u_wsindy.loc[nu_vec][j].to_numpy()
            semui = std_u_wsindy.loc[nu_vec][j].to_numpy() / np.sqrt(reps)
            dfs_mean[i][f'u{j}_WSINDY'] = meanui
            dfs_std[i][f'u{j}_WSINDY'] = stdui
            dfs_sem[i][f'u{j}_WSINDY'] = semui

        # dfs_mean[i][dfs_mean[i] > 1] = 1

# %% [markdown]

# # Plot errors

# %%

if len(dfs_mean) == 1:

    # Coefficient Error (sem)
    cols_c = ['socp_sm', 'lasso', 'WSINDY']
    labels_c = {'socp_sm': 'DSINDy', 'lasso': r'$\ell_1$-SINDy', 'WSINDY': 'WSINDy'}
    gen_plot_set(cols_c, dfs_mean[0], dfs_sem[0], labels=labels_c,
                 fn=f'{base_dir}/{base_dir}_coef_summary_err', error_type='c')

    cols_c_socp = ['socp_sm', 'socp_theory']
    labels_c_socp = {'socp_sm': 'DSINDy (Pareto)',
                     'socp_theory': 'DSINDy (theory)'}
    gen_plot_set(cols_c_socp, dfs_mean[0], dfs_sem[0], labels=labels_c_socp,
                 fn=f'{base_dir}/{base_dir}_coef_socp_summary_err',
                 error_type='c', colors=[colors[0], colors[3]])

    # Reconstruction Error (sem)
    cols_u = ['pred_socp_sm', 'pred_lasso', 'WSINDY']
    labels_u = {'pred_socp_sm': 'DSINDy', 'pred_lasso': r'$\ell_1$-SINDy',
                'WSINDY': 'WSINDy'}
    gen_plot_set(cols_u, dfs_mean[0], dfs_sem[0], labels=labels_u,
                 fn=f'{base_dir}/{base_dir}_u_summary_err',
                 error_type='u')

    # Reconstruction Error (sem)
    cols_u_socp = ['pred_socp_sm', 'pred_socp_theory']
    labels_u_socp = {'pred_socp_sm': 'DSINDy (Pareto)',
                'pred_socp_theory': 'DSINDy (theory)'}
    gen_plot_set(cols_u_socp, dfs_mean[0], dfs_sem[0], labels=labels_u_socp,
                 fn=f'{base_dir}/{base_dir}_u_socp_summary_err',
                 error_type='u', colors=[colors[0], colors[3]])


    # Derivative Error (std)
    cols_du = ['socp_sm', 'tikreg']
    labels_du = {'socp_sm': 'DSINDy', 'tikreg': 'Tikhonov Regularization'}
    gen_plot_set(cols_du, dfs_mean[0], dfs_std[0],
                 ylabel_start=f'{SysName}\n\n', labels=labels_du,
                 fn=f'{base_dir}/{base_dir}_du_summary_err',
                 error_type='du')

    # Smoothing Error (std)
    cols_smooth = ['smooth_proj', 'smooth_GP']
    labels_smooth = {'smooth_proj': 'IterPSDN', 'smooth_GP': 'GP'}
    gen_plot_set(cols_smooth, dfs_mean[0], dfs_std[0],
                 ylabel_start=f'{SysName}\n\n', labels=labels_smooth,
                 fn=f'{base_dir}/{base_dir}_smooth_summary_err',
                 error_type='us')

else:

    # # Smoothing error
    # for i in range(1, m + 1):
    #     cols_smooth = [f'u{i}_smooth_proj', f'u{i}_smooth_GP']
    #     labels_smooth = {f'u{i}_smooth_proj': 'PSDN',
    #                      f'u{i}_smooth_GP': 'GP'}
    #     gen_plot(cols_smooth, dfs_mean, dfs_std,
    #              r'Relative $\tilde{\bm{u}}_' + f'{i}' + r'$ error', N_vec,
    #              labels=labels_smooth,
    #              fn=f'{base_dir}/{base_dir}_u{i}_smooth_err')

    # Coefficient error
    for i in range(1, m + 1):
        # Columns for coefficient errors
        if compare_with_wsindy:
            cols_c = [f'c{i}_socp_sm', f'c{i}_lasso', f'c{i}_WSINDY']
            labels_c = {f'c{i}_socp_sm': 'DSINDy',
                        f'c{i}_lasso': r'$\ell_1$-SINDy',
                        f'c{i}_WSINDY': 'WSINDy'}
        else:
            cols_c = [f'c{i}_socp_sm', f'c{i}_lasso']
            labels_c = {f'c{i}_socp_sm': 'DSINDy',
                        f'c{i}_lasso': r'$\ell_1$-SINDy'}

        cols_socp_c = [f'c{i}_socp_sm', f'c{i}_socp_theory']
        labels_socp_c = {f'c{i}_socp_sm': 'DSINDy (Pareto)',
                         f'c{i}_socp_theory': 'DSINDy (theory)'}

        # Coefficient Error
        gen_plot(cols_c, dfs_mean, dfs_sem,
                 r'Relative $\bm{c}_' + f'{i}' + r'$ error',
                 N_vec, labels=labels_c,
                 fn=f'{base_dir}/{base_dir}_c{i}_err')

        # Coefficient Error SOCP
        gen_plot(cols_socp_c, dfs_mean, dfs_sem,
                 r'Relative $\bm{c}_' + f'{i}' + r'$ error',
                 N_vec, labels=labels_socp_c,
                 fn=f'{base_dir}/{base_dir}_c{i}_socp_err',
                 colors=[colors[0], colors[3]])

    # Derivative error
    for i in range(1, m + 1):
        cols_du = [f'du{i}_socp_sm', f'du{i}_tikreg']
        labels_du = {f'du{i}_socp_sm': 'DSINDy',
                     f'du{i}_tikreg': 'Tikhonov Regularization'}
        gen_plot(cols_du, dfs_mean, dfs_std,
                 r'Relative $\dot{\bm{u}}_' + f'{i}' + r'$ error',
                 N_vec, labels=labels_du,
                 fn=f'{base_dir}/{base_dir}_u{i}_dot_err')

    # Training + testing prediciton error
    for i in range(1, m + 1):
        cols_u = [f'u{i}_pred_socp_sm', f'u{i}_pred_lasso', f'u{i}_WSINDY']
        labels_u = {f'u{i}_pred_socp_sm': 'DSINDy',
                    f'u{i}_pred_socp_GP': 'SOCP-GP',
                    f'u{i}_pred_lasso': r'$\ell_1$-SINDy',
                    f'u{i}_pred_lasso_GP': 'Lasso-GP',
                    f'u{i}_WSINDY': 'WSINDy'}
        gen_plot(cols_u, dfs_mean, dfs_sem,
                 r'Relative $\bm{u}_' + f'{i}' + r'$ error', N_vec,
                 labels=labels_u,
                 fn=f'{base_dir}/{base_dir}_u{i}_err')

        cols_socp_u = [f'u{i}_pred_socp_sm',
                       f'u{i}_pred_socp_theory']
        labels_socp_u = {f'u{i}_pred_socp_es': 'DSINDy (Pareto)',
                         f'u{i}_pred_socp_sm': 'DSINDy (Pareto)',
                         f'u{i}_pred_socp_theory': 'DSINDy (theory)'}
        gen_plot(cols_socp_u, dfs_mean, dfs_sem,
                 r'Relative $\bm{u}_' + f'{i}' + r'$ error', N_vec,
                 labels=labels_socp_u,
                 fn=f'{base_dir}/{base_dir}_u{i}_socp_err',
                 colors=[colors[0], colors[3]])

# Failed solution
cols_fail = ['socp_failed', 'lasso_failed', 'wsindy_failed']
labels_fail = {'lasso_failed': r'$\ell_1$-SINDy',
               'socp_failed': 'DSINDy',
               'wsindy_failed': 'WSINDy',
               'socp_theory_failed': 'DSINDy (theory)'}
gen_plot(cols_fail, dfs_mean, dfs_std, 'Failure rate', N_vec,
         labels=labels_fail, fn=f'{base_dir}/{base_dir}_failed',
         logy=False, include_std=False, ypad=2)

# Failed solution
cols_fail = ['socp_failed', 'socp_theory_failed']
labels_fail = {'lasso_failed': r'$\ell_1$-SINDy',
               'socp_failed': 'DSINDy',
               'wsindy_failed': 'WSINDy',
               'socp_theory_failed': 'DSINDy (theory)'}
gen_plot(cols_fail, dfs_mean, dfs_std, 'Failure rate', N_vec,
         labels=labels_fail, fn=f'{base_dir}/{base_dir}_socp_failed',
         logy=False, include_std=False, ypad=2,
         colors=[colors[0], colors[3]])
