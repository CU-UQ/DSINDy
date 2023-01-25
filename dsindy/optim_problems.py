"""Functions for running optimization problems."""

import numpy as np
import numpy.linalg as la
import sklearn.linear_model as lm
from cvxopt import matrix, solvers
from mosek import iparam, dparam
# import importlib
# import time

import dsindy.L_curve_utils_lasso as lcu
import dsindy.utils as utils


def reset_mosek_params():
    """Run to reset optimization parameters."""
    solvers.options['mosek'] = {
        iparam.log: 0,
        dparam.intpnt_co_tol_pfeas: 1e-10,
        dparam.intpnt_co_tol_dfeas: 1e-10,
        dparam.intpnt_co_tol_infeas: 1e-10
    }


def increase_mosek_tol():
    """Increase mosek tolerance."""
    old_ptol = solvers.options['mosek'][dparam.intpnt_co_tol_pfeas]
    old_dtol = solvers.options['mosek'][dparam.intpnt_co_tol_dfeas]
    old_infeas = solvers.options['mosek'][dparam.intpnt_co_tol_infeas]
    solvers.options['mosek'] = {
        iparam.log: 0,
        dparam.intpnt_co_tol_pfeas: old_ptol * 10,
        dparam.intpnt_co_tol_dfeas: old_dtol * 10,
        dparam.intpnt_co_tol_infeas: old_infeas * 10
    }


def deriv_tik_reg(t,
                  u,
                  udot,
                  title='',
                  plot=False,
                  alpha=None,
                  opt_params=None):
    """Approximate derivative from states using Tikhonov regularization."""
    A = utils.get_discrete_integral_matrix(t)
    D = utils.get_derivative_matrix(t)

    if alpha is None:
        alpha, x_final, y_final = lcu.findAlphaMaxCurve(A,
                                                        u,
                                                        title,
                                                        plot_results=plot,
                                                        D=D,
                                                        method='2',
                                                        opt_params=opt_params)

    deriv = np.linalg.inv(A.T @ A + alpha * D.T @ D) @ A.T @ u

    return deriv


def run_socp_optimization(u, A, B, D, W, args, opt_params=None, start=None):
    """Iteratively run SOCP based optimization."""
    B_new = np.copy(B)
    p = np.size(W, 0)
    Dc = np.eye(p)
    c_old = np.ones(p)
    opt_params = opt_params.copy()
    stop_iterations = False
    for i in range(opt_params['max_IRW_iter']):
        for j in range(5):
            try:
                alpha = lcu.findAlphaMaxCurve(A,
                                              u,
                                              f'SOCP L curve: Iteration {i+1}',
                                              method='SOCP',
                                              D=D,
                                              B=B_new,
                                              y2=args,
                                              plot_results=True,
                                              opt_params=opt_params)[0]
                break
            except Exception as ex:

                # Convert exception to dictionary
                ex = eval(str(ex))

                if str(ex['name']) == 'Too large':
                    a_max_new = opt_params['a_max'] / 2
                    # a_max_new = ex['alpha'] / 1.1
                    print(f'Decreasing a_max to {a_max_new}')
                    opt_params['a_max'] = a_max_new
                if str(ex['name']) == 'Too small':
                    a_min_new = opt_params['a_min'] * 2
                    # a_min_new = ex['alpha'] * 1.1
                    print(f'Increasing a_min to {a_min_new}')
                    opt_params['a_min'] = a_min_new
                # if str(ex) == 'Both':
                #     a_max_new = opt_params['a_max'] / 2
                #     print(f'Decreasing a_max to {a_max_new}')
                #     opt_params['a_max'] = a_max_new
                #     a_min_new = opt_params['a_min'] * 2
                #     print(f'Increasing a_min to {a_min_new}')
                #     opt_params['a_min'] = a_min_new
            if opt_params['a_min'] > opt_params['a_max']:
                print('amin > amax: breaking')
                stop_iterations = True
                break
        if stop_iterations:
            break

        x = solve_socp(u, args, A, B_new, D, alpha)[0]
        cW = np.hstack((np.zeros(p).reshape(-1, 1), B)) @ x
        coef_change = la.norm(W @ cW - c_old) / la.norm(c_old)
        print(f'Change in coefs at iteration {i+1}: {coef_change:.4g}')
        c_old = W @ cW
        if coef_change < 1e-4:
            break
        if np.max(np.abs(cW)) < 1e-6:
            break
        Dc = np.diag(1 / (np.abs(cW) + 1e-4 * np.max(np.abs(cW))))
        B_new = Dc @ B
    print('Final coefs:')
    print(c_old)
    return (x)


def run_weighted_lasso(A,
                       y,
                       W,
                       method='1',
                       show_L_curve=False,
                       type='monomial',
                       species='u1',
                       opt_params=None):
    """Iteratively run reweighted lasso."""
    W2_inv = np.eye(np.size(A, 1))
    c_old = np.ones(np.size(A, 1))
    for i in range(opt_params['max_IRW_iter']):
        alpha_final = lcu.findAlphaMaxCurve(
            A @ W2_inv,
            y,
            f'L curve for {type} system ({species}).',
            plot_results=show_L_curve,
            method='1',
            opt_params=opt_params)[0]
        las = lm.Lasso(alpha=alpha_final,
                       fit_intercept=False,
                       tol=opt_params['tol'],
                       max_iter=opt_params['max_iter'])
        las.fit(A @ W2_inv, y)
        cW = W2_inv @ las.coef_
        coef_change = la.norm(W @ cW - c_old) / la.norm(c_old)
        print(f'Change in coefs at iteration {i+1}: {coef_change:.4g}')
        c_old = W @ cW
        if coef_change < 1e-4:
            break

        W2 = np.diag(1 / (np.abs(cW) + 1e-4 * np.max(np.abs(cW))))
        W2_inv = la.inv(W2)
        # Only show L-curve for first iteration
        show_L_curve = False
    print('Final coefs:')
    print(c_old)
    return (c_old)


def solve_socp(y,
               y2,
               A,
               B,
               D,
               alpha,
               B_subs=None,
               P_subs=None,
               return_sol=False,
               solver=None,
               x_old=None,
               checkResid=True):
    """Solve SOCP to learn state derivatives.

    Solves the following:
        minimize ||Bx||_1
        subject to ||Dx||_2 < alpha
                   ||Ax - y||_2 < sigma

    Args:
        y (type): description

    Returns:
        type: x
        type: ||Bx||_1
        type: ||Dx||_2
    """
    reset_mosek_params()
    # print(alpha)
    # print(alpha)
    m = np.size(A, 1)
    N = np.size(A, 0)
    sB = np.size(B, 0)
    sD = np.size(D, 0)

    c = matrix(np.hstack((np.zeros(m), np.ones(sB))))
    Gl = matrix(
        np.vstack((np.hstack((np.zeros(sB).reshape(-1, 1), B, -np.eye(sB))),
                   np.hstack((np.zeros(sB).reshape(-1, 1), -B, -np.eye(sB))))))
    hl = matrix(np.zeros(2 * sB))

    # ||D udot|| < C
    Gq1 = matrix(
        np.vstack((np.zeros(m + sB),
                   np.hstack((np.zeros(sD).reshape(-1, 1), D, np.zeros(
                       (sD, sB)))))))
    hq1 = matrix(np.hstack((np.sqrt(N) * y2, np.zeros(sD))))
    # hq1 = matrix(np.hstack((np.sqrt(N) * alpha, np.zeros(sD))))

    # ||[1 A]udot - utilde|| < alpha
    Gq2 = matrix(
        np.vstack((np.zeros(m + sB), np.hstack((A, np.zeros((N, sB)))))))
    hq2 = matrix(np.hstack((np.sqrt(N) * alpha, y)))
    # hq2 = matrix(np.hstack((np.sqrt(N) * y2, y)))

    # startTime = time.time()
    # executionTime = (time.time() - startTime)
    # print('Execution time for mosek solver: ' + str(executionTime))

    for i in range(10):

        sol = solvers.socp(c, Gl, hl, [Gq1, Gq2], [hq1, hq2], solver='mosek')
        # executionTime = (time.time() - startTime)
        # print('Execution time for mosek solver: ' + str(executionTime))

        # Check if optimization succeeded
        # print(sol['x'])
        # print(sol)
        if sol['x'] is not None:
            if return_sol:
                return (sol)

            x = np.array(sol['x']).flatten()[:m]
            resres = np.linalg.norm(
                np.hstack((np.zeros(sB).reshape(-1, 1), B)) @ x, 1)
            solres = np.linalg.norm(A @ x - y, 2)
            # solres = np.linalg.norm(D @ x[1:], 2)

            # If the regularization residual is too small, raise exception
            if checkResid:
                if resres < 1e-6:
                    ex_dict = {'name': 'Too large', 'alpha': alpha}
                    print(f'Current alpha: {alpha}')
                    raise Exception(ex_dict)

                if resres > 1e8:
                    ex_dict = {'name': 'Too small', 'alpha': alpha}
                    print(f'Current alpha: {alpha}')
                    raise Exception(ex_dict)

            return (x, solres, resres)

        # If optimization has not suceedeed, change bounds on both max/min
        # alpha.
        if i < 9:
            increase_mosek_tol()
            print('Increasing mosek tolerence')
        else:
            ex_dict = {'name': 'Too small', 'alpha': alpha}
            print(f'Current alpha: {alpha}')
            raise Exception(ex_dict)
