"""Contains functions for finding corner of Paerto curve."""

import numpy as np
import sklearn.linear_model as lm
import plotly.graph_objects as go

import eqndiscov.optim_problems as op


def find_curvature(x_temp, y_temp, extrema):
    """Find curvature based on three points.

    Given three points with x cordinates and y coordinates in x_temp and
    y_temp, respectively, calculate the normalized log curvature. The
    normalization is done using values in extrema=[xmin,ymin,xmax,ymax].
    """
    # Change to a log scale and normalize on [0,1]
    x = (np.log10(x_temp) - np.log10(extrema[0])) / (
        np.log10(extrema[2]) - np.log10(extrema[0])
        )
    y = (np.log10(y_temp) - np.log10(extrema[1])) / (
        np.log10(extrema[3]) - np.log10(extrema[1])
        )

    # If points are on straight line return error
    if (x[0] == x[1] and x[1] == x[2]):
        print('x equal')
        return('err')
    if (y[0] == y[1] and y[1] == y[2]):
        print('y equal')
        return('err')

    # Calculate curvature
    d1 = np.sqrt((y[1] - y[0])**2 + (x[1] - x[0])**2)
    d2 = np.sqrt((y[2] - y[0])**2 + (x[2] - x[0])**2)
    d3 = np.sqrt((y[1] - y[2])**2 + (x[1] - x[2])**2)
    term1 = (x[1] - x[0]) * (y[2] - y[1])
    term2 = (y[1] - y[0]) * (x[2] - x[1])
    area = 1 / 2 * (term1 - term2)
    curvature = 4 * area / (d1 * d2 * d3)

    return(curvature)


def find_alpha_vec(A, y, method, D=None, B=None, y2=None, opt_params=None):
    """Find initial four values of alpha."""
    # Initiate vectors to store the residuals
    sol_res = np.zeros(4)
    reg_res = np.zeros(4)

    a_min = opt_params['a_min']
    a_max = opt_params['a_max']
    # Calculate the least squares solution and residual
    if B is None:
        ls_sol = np.linalg.pinv(A.T @ A) @ A.T @ y
        ls_res = np.linalg.norm(A @ ls_sol - y)
    else:
        if method == 'SOCP':
            ls_res = 0

    # Golden ratio: Used to search hyperparameter space
    pGS = (1 + np.sqrt(5)) / 2

    # Update a_min and a_max so that the minimum regularization
    # residual does not equal zero and the minimum solution residual does not
    # equal the least squares solution. This helps the algorithm converge
    # properly.
    while np.min(reg_res) == 0 or np.min(sol_res - ls_res) < 1e-12:

        # Specify the two interior alpha values using the golden ratio.
        alpha2 = a_max**(1 / (1 + pGS)) * a_min**(pGS / (1 + pGS))
        alpha3 = a_max**(pGS / (1 + pGS)) * a_min**(1 / (1 + pGS))
        # Construct vector containing the 4 alpha values
        alpha_vec = np.array([a_min, alpha2, alpha3, a_max])
        # Perform lasso at the 4 alpha values (based on method)
        for i in range(4):
            if method == '1':
                las = lm.Lasso(alpha=alpha_vec[i], fit_intercept=False,
                               tol=opt_params['tol'],
                               max_iter=opt_params['max_iter'])
                las.fit(A, y)
                sol_res[i] = np.linalg.norm(A @ las.coef_ - y)
                reg_res[i] = np.sum(np.abs(las.coef_))
            elif method == '2':
                coef = np.linalg.inv(
                    A.T @ A + alpha_vec[i] * D.T @ D
                    ) @ A.T @ y
                sol_res[i] = np.linalg.norm(A @ coef - y)
                reg_res[i] = np.linalg.norm(D @ coef)
            elif method == 'SOCP':
                sol_res[i], reg_res[i] = op.solve_socp(
                    y, y2, A, B, D, alpha_vec[i]
                    )[1:]

        if np.min(reg_res) == 0:
            a_max = a_max / 2
        if np.min(sol_res - ls_res) < 1e-12:
            a_min = a_min * 2

    # Return the final alpha_vec and residuals
    return alpha_vec, sol_res, reg_res


def lassoLCurve(A, y, D=None, B=None,
                method='2', max_iter=100000, y2=None,
                opt_params=None):
    """Run L curve algorithm to find hyperparameter alpha.

    Three methods are possible:
    -method='1': ||Ax-y||_2 + alpha||x||_1
    -method='2': ||Ax-y||_2 + alpha||Dx||_2
    -method='SOCP'

    """
    # Golden ratio: Used to search hyperparameter space
    pGS = (1 + np.sqrt(5)) / 2

    # Calculate Gramians to speed up calculation
    if method == '2':
        GA = A.T @ A
        if D is not None:
            DA = D.T @ D

    for iter in range(max_iter):

        # On first iteration find initial alpha vec and corresponding extrema
        if iter == 0:
            alpha_vec, sol_res, reg_res = find_alpha_vec(
                A, y, method, D=D, B=B,
                y2=y2, opt_params=opt_params)
            alpha_all = np.copy(alpha_vec)
            sol_res_all = np.copy(sol_res)
            reg_res_all = np.copy(reg_res)

            extrema = np.array([
                np.min(sol_res), np.min(reg_res),
                np.max(sol_res), np.max(reg_res)])

        # Set a_min and a_max values
        a_min = alpha_vec[0]
        a_max = alpha_vec[3]

        # Find curvature using the solution and regularization residuals
        curvature1 = find_curvature(sol_res[:3], reg_res[:3], extrema)
        curvature2 = find_curvature(sol_res[1:], reg_res[1:], extrema)

        # If error results calculate new alpha_vec
        while curvature1 == 'err' or curvature2 == 'err':
            print('Curvature 1: ' + str(curvature1))
            print('Curvature 2: ' + str(curvature2))
            if curvature1 == 'err':
                a_min = a_min * 2
            if curvature2 == 'err':
                a_max = a_max / 2
            if a_max < a_min:
                print('error: alpha max less than alpha min')
                print(alpha_vec)
                return alpha_vec[0]
            alpha_vec, sol_res, reg_res = find_alpha_vec(
                A, y, method, D=D, B=B,
                y2=y2, opt_params=opt_params)
            a_min = alpha_vec[0]
            a_max = alpha_vec[3]
            curvature1 = find_curvature(sol_res[:3], reg_res[:3], extrema)
            curvature2 = find_curvature(sol_res[1:], reg_res[1:], extrema)

        # Once alpha_vec has converged break and return value
        if (alpha_vec[3] - alpha_vec[0]) / alpha_vec[0] < 1e-2:
            if curvature1 > curvature2:
                alpha_final = alpha_vec[1]
            else:
                alpha_final = alpha_vec[2]
            break

        # Best on region of maximum curvature, modify alpha_vec
        if curvature1 > curvature2:
            alpha_vec[2:4] = alpha_vec[1:3]
            sol_res[2:4] = sol_res[1:3]
            reg_res[2:4] = reg_res[1:3]
            new_alpha1 = alpha_vec[3]**(1 / (1 + pGS))
            new_alpha2 = alpha_vec[0]**(pGS / (1 + pGS))
            alpha_vec[1] = new_alpha1 * new_alpha2
            lasso_idx = 1
        else:
            alpha_vec[0:2] = alpha_vec[1:3]
            sol_res[0:2] = sol_res[1:3]
            reg_res[0:2] = reg_res[1:3]
            new_alpha1 = alpha_vec[0]**(1 / (1 + pGS))
            new_alpha2 = alpha_vec[3]**(pGS / (1 + pGS))
            alpha_vec[2] = new_alpha1 * new_alpha2
            lasso_idx = 2

        # For new alpha, perform lasso based on the necessary method
        if method == '1':
            las = lm.Lasso(alpha=alpha_vec[lasso_idx], fit_intercept=False,
                           tol=opt_params['tol'],
                           max_iter=opt_params['max_iter'])
            las.fit(A, y)
            sol_res[lasso_idx] = np.linalg.norm(A @ las.coef_ - y)
            reg_res[lasso_idx] = np.sum(np.abs(D @ las.coef_))
        elif method == '2':
            coef = np.linalg.inv(GA + alpha_vec[lasso_idx] * DA) @ A.T @ y
            sol_res[lasso_idx] = np.linalg.norm(A @ coef - y)
            reg_res[lasso_idx] = np.linalg.norm(D @ coef)
        elif method == 'SOCP':
            sol_res[lasso_idx], reg_res[lasso_idx] = op.solve_socp(
                y, y2, A, B, D, alpha_vec[lasso_idx]
                )[1:]

        alpha_all = np.append(alpha_all, alpha_vec[lasso_idx])
        sol_res_all = np.append(sol_res_all, sol_res[lasso_idx])
        reg_res_all = np.append(reg_res_all, reg_res[lasso_idx])

    return alpha_final, alpha_all, sol_res_all, reg_res_all


def get_L_curve_data(A, y, method='1', D=None,
                     B=None, c_actual=None, tol=1e-12, y2=None,
                     opt_params=None):
    """Gives tsolution and regularization residuals to plot L-curve.

    Three optimzation methods are possible:
    -method='1': ||Ax-y||_2 + alpha||x||_1
    -method='2': ||Ax-y||_2 + alpha||Dx||_2
    -method='SOCP'
    """
    # Generate list of alphas at which to perform lasso
    a_min = opt_params['a_min']
    a_max = opt_params['a_max']
    a_num = opt_params['a_num']
    alphas = np.logspace(np.log10(a_min), np.log10(a_max), a_num)

    norm_min = 10000
    reg_res = np.zeros(a_num)
    sol_res = np.zeros(a_num)
    for k in range(a_num):
        a = alphas[k]
        if method == '1':
            las = lm.Lasso(alpha=a, fit_intercept=False, tol=opt_params['tol'],
                           max_iter=opt_params['max_iter'])
            las.fit(A, y)
            x = las.coef_
            reg_res[k] = np.sum(np.abs(x))
            sol_res[k] = np.linalg.norm(A @ x - y)
        elif method == '2':
            x = np.linalg.inv(A.T @ A + a * D.T @ D) @ np.transpose(A) @ y
            reg_res[k] = np.linalg.norm(D @ x)
            sol_res[k] = np.linalg.norm(A @ x - y)
        elif method == 'SOCP':
            sol_res[k], reg_res[k] = op.solve_socp(
                y, y2, A, B, D, a)[1:]

        # Check if current alpha is optimal signal
        if c_actual is not None:
            norm_cur = np.linalg.norm(x - c_actual)
            if norm_cur < norm_min:
                alpha_best = a
                norm_min = norm_cur
                x_best = x

    if c_actual is None:
        return(reg_res, sol_res, alphas)
    else:
        return(reg_res, sol_res, alphas, alpha_best, x_best)


def findAlphaMaxCurve(A, y, title, plot_results=False,
                      D=None, B=None, method='1', y2=None, opt_params=None):
    """Find alpha_final using the L curve algorithm.

    Three optimzation methods are possible:
    -method='1': ||Ax-y||_2 + alpha||Dx||_1
    -method='2': ||Ax-y||_2 + alpha||Dx||_2
    -method='SOCP'

    """
    if D is None:
        D = np.eye(np.size(A, 1))

    alpha_final, alpha_all, sol_res_all, reg_res_all = lassoLCurve(
        A, y, D=D, method=method, B=B,
        y2=y2, opt_params=opt_params
        )

    if method == '1':
        las = lm.Lasso(alpha=alpha_final, fit_intercept=False,
                       tol=opt_params['tol'],
                       max_iter=opt_params['max_iter'])
        las.fit(A, y)
        sol_res = np.linalg.norm(A @ las.coef_ - y)
        reg_res = np.sum(np.abs(D @ las.coef_))
    elif method == '2':
        coef = np.linalg.inv(
            np.transpose(A) @ A + alpha_final * np.transpose(D) @ D
            ) @ np.transpose(A) @ y
        sol_res = np.linalg.norm(A @ coef - y)
        reg_res = np.linalg.norm(D @ coef)
    elif method == 'SOCP':
        sol_res, reg_res = op.solve_socp(y, y2, A, B, D, alpha_final)[1:]

    if plot_results:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=reg_res_all, y=sol_res_all,
                                 text=alpha_all, mode='markers'))
        fig.add_trace(go.Scatter(x=[reg_res], y=[sol_res],
                                 text=alpha_final))
        fig.update_xaxes(type='log',
                         title_text='Regularization residual (l1 or l2)')
        fig.update_yaxes(type='log',
                         title_text='Solution residual (l2)')
        fig.update_layout(title_text=title, width=500, height=350,
                          showlegend=False)
        fig.show()

    return(alpha_final, reg_res, sol_res)
