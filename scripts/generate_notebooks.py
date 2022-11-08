"""Script to generate a set of SOCP and Lasso optimizations."""
import os
import json
import subprocess


def run_notebook(file_name, out_file_type='html', **arguments):
    """Run and output notebook for a given set of parameters."""
    realization = arguments['realization']
    nu = arguments['nu']
    system = arguments['system']
    ttrain = arguments['ttrain']
    N = arguments['N']

    base_name = f'system={system}_nu={nu}_N={N}_ttrain={ttrain}'
    file_name_out = f'notebook_{base_name}_realization={realization}'

    out_dir = f'/app/current_output/{base_name}/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_name_final = file_name_out + '_' + str(0)
    i = 1
    while os.path.exists(out_dir + file_name_final + '.' + out_file_type):
        file_name_final = file_name_out + '_' + str(i)
        i = i + 1

    # Set up arguments file
    with open('/app/arguments.json', 'w') as fid:
        json.dump(arguments, fid)
    #
    # Generate notebook
    subprocess.call([
        'jupytext',
        '--from', 'py:percent',
        '--to', 'notebook',
        '/app/notebooks/' + file_name + '.py'
        ])

    # Run notebook
    # In this call is output directory relative to input file location?
    subprocess.call([
        'jupyter', 'nbconvert', '/app/notebooks/' + file_name + '.ipynb',
        '--execute',
        '--to', out_file_type,
        '--output', out_dir + '/' + file_name_final,
        '--TagRemovePreprocessor.enabled', 'True',
        '--TagRemovePreprocessor.remove_input_tags', 'remove_input',
        '--TagRemovePreprocessor.remove_cell_tags', 'remove_cell'
        ])


def run_notebook_set(system, N_vec,
                     ttrain=10,
                     nu_vec=[1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1],
                     n_realizations=10, start=0):
    """Generate a set of notebooks at different N and nu."""
    if system[0] == '2':
        datadir = '/app/paper_noise_realizations/Duffing/'
    if system[0] == '3':
        datadir = '/app/paper_noise_realizations/Van_der_Pol/'
    if system[0] == '4':
        datadir = '/app/paper_noise_realizations/Rossler/'

    for N in N_vec:
        for nu in nu_vec:
            for i in range(n_realizations):
                run_notebook('run_SOCP_and_IRW_Lasso',
                             realization=start + i,
                             nu=nu,
                             system=system,
                             ttrain=ttrain,
                             N=N,
                             datadir=datadir)


# system = '2a'
# N_vec = [1000]
# nu_vec = [0.0001]
# run_notebook_set(system, N_vec, start=25, n_realizations=1, nu_vec=nu_vec)

system = '2b'
N_vec = [8000]
# nu_vec = [1]
# run_notebook_set(system, N_vec, start=2, n_realizations=28, nu_vec=nu_vec)

nu_vec = [0.0001]
run_notebook_set(system, N_vec, start=0, n_realizations=30, nu_vec=nu_vec)

# system = '2c'
# N_vec = [1000]
# run_notebook_set(system, N_vec, ttrain=30, start=10)

# system = '3'
# N_vec = [1000]
# run_notebook_set(system, N_vec, start=10, n_realizations=20)

# system = '4'
# N_vec = [1000]
# run_notebook_set(system, N_vec, start=10, n_realizations=20)
