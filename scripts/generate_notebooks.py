"""Script to generate a set of SOCP and Lasso optimizations."""
import os
import json
import subprocess


def run_notebook(file_name,
                 out_file_type='html',
                 bdir='/home/jacqui/projects/DSINDy/',
                 **arguments):
    """Run and output notebook for a given set of parameters."""
    realization = arguments['realization']
    nu = arguments['nu']
    system = arguments['system']
    ttrain = arguments['ttrain']
    N = arguments['N']

    base_name = f'system={system}_nu={nu}_N={N}_ttrain={ttrain}'
    file_name_out = f'notebook_{base_name}_realization={realization}'

    out_dir = f'{bdir}/current_output/{base_name}/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_name_final = file_name_out + '_' + str(0)
    i = 1
    while os.path.exists(out_dir + file_name_final + '.' + out_file_type):
        file_name_final = file_name_out + '_' + str(i)
        i = i + 1

    # Set up arguments file
    with open(f'{bdir}/arguments.json', 'w') as fid:
        json.dump(arguments, fid)

    # Generate notebook
    subprocess.call([
        'jupytext', '--from', 'py:percent', '--to', 'notebook',
        f'{bdir}/notebooks/{file_name}.py'
    ])

    # Run notebook
    # In this call is output directory relative to input file location?
    subprocess.call([
        'jupyter', 'nbconvert', f'{bdir}/notebooks/{file_name}.ipynb',
        '--execute', '--to', out_file_type, '--output',
        f'{out_dir}/{file_name_final}', '--TagRemovePreprocessor.enabled',
        'True', '--TagRemovePreprocessor.remove_input_tags', 'remove_input',
        '--TagRemovePreprocessor.remove_cell_tags', 'remove_cell'
    ])

    # Cleanup
    os.remove(f'{bdir}/notebooks/{file_name}.ipynb')


def run_notebook_set(system,
                     N_vec,
                     ttrain=10,
                     nu_vec=[1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1],
                     n_realizations=10,
                     start=0,
                     bdir='/home/jacqui/projects/DSINDy/'):
    """Generate a set of notebooks at different N and nu."""
    if system[0] == '2':
        datadir = f'{bdir}/paper_noise_realizations/Duffing/'
    if system[0] == '3':
        datadir = f'{bdir}/paper_noise_realizations/Van_der_Pol/'
    if system[0] == '4':
        datadir = f'{bdir}/paper_noise_realizations/Rossler/'
    if system[0] == '5':
        datadir = f'{bdir}/paper_noise_realizations/Lorenz_96/'

    for N in N_vec:
        for nu in nu_vec:
            for i in range(n_realizations):
                run_notebook('run_SOCP_and_IRW_Lasso',
                             realization=start + i,
                             nu=nu,
                             system=system,
                             ttrain=ttrain,
                             N=N,
                             datadir=datadir,
                             bdir=bdir)


system = '2b'
N_vec = [1000]
nu_vec = [1]
run_notebook_set(system, N_vec, start=25, n_realizations=1, nu_vec=nu_vec)

# system = '2b'
# N_vec = [250]
# nu_vec = [0.01]
# run_notebook_set(system,
#                  N_vec,
#                  bdir='/home/jacqui/DSINDy',
#                  start=11,
#                  n_realizations=1,
#                  nu_vec=nu_vec)

# system = '2c'
# N_vec = [1000]
# run_notebook_set(system, N_vec, ttrain=30, start=10)

# system = '3'
# N_vec = [1000]
# run_notebook_set(system, N_vec, start=10, n_realizations=20)

# system = '4'
# N_vec = [1000]
# run_notebook_set(system, N_vec, start=10, n_realizations=20)

# system = '5'
# N_vec = [2000]
# ttrain = 5
# run_notebook_set(system,
#                  N_vec,
#                  start=0,
#                  n_realizations=19,
#                  ttrain=ttrain,
#                  nu_vec=[.1])

print('Code finished!')
