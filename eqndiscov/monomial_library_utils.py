"""Contains functions needed to make the monomial libraries."""

import numpy as np
import scipy.special as sps
import itertools

from eqndiscov.utils import ncr


def make_Theta(u, d=3):
    """Create the library of monomials up to degree d."""
    m = np.size(u, 0)
    Theta_t = np.ones(np.size(u[0]))
    mi_mat = make_mi_mat(m, d)
    for mi in mi_mat:
        if np.sum(mi) == 0:
            continue
        Theta_t = np.vstack((Theta_t, np.prod((u.T**mi).T, axis=0)))
    Theta = np.transpose(Theta_t)
    return Theta


def center_Theta(Theta, d, m, nu):
    """Center library Theta so that each element is unbiased."""
    mi_mat = make_mi_mat(m, d)
    Theta_shifted = np.ones((np.size(Theta, 0), np.size(Theta, 1)))

    # Set noise vector depending on whether nu is scalar or vector
    if np.size(nu) == 1:
        nu_vec = nu * np.ones(m)
    else:
        nu_vec = np.copy(nu)

    # Iterate through basis
    for i, K_vec in enumerate(mi_mat):
        total_sum = np.zeros(np.size(Theta, 0))

        # Calculate the indice combinations in the summation
        ranges = []
        for K in K_vec:
            ranges.append(list(range(K + 1)))
        combo_list = itertools.product(*ranges)

        # Iterate through each indice combination
        for combo in combo_list:
            k_vec = np.array(combo)
            should_continue = False
            if sum(k_vec) == 0:
                continue

            # Continue if order is odd since E[epsilon^(odd)]=0
            for k in k_vec:
                if k % 2 != 0 and k != 0:
                    should_continue = True
            if should_continue:
                continue

            # Add to the needed adjustment term
            const = 1
            exp_eps = 1
            for ii, k in enumerate(k_vec):
                const *= ncr(K_vec[ii], k)
                exp_eps *= np.prod(range(k - 1, 0, -2)) * nu_vec[ii]**(k / 2)

            j = np.where(np.all(mi_mat == K_vec - k_vec, axis=1))[0][0]

            total_sum += const * exp_eps * Theta_shifted[:, j]

        Theta_shifted[:, i] = Theta[:, i] - total_sum

    return (Theta_shifted)


def make_M2(w_noise, w_tilde, d, nu):
    r"""Calculate expected moments in $(\Theta(\tilde{w}))^T \Theta(w)$.

    Args:
        w_noise (d X N np.array): N noisy measurements of the d states.
        w_tilde (d X N np.array): N smoothed measurements of the d states.
        d (int): maximum degree for the returned moments.
        nu (float or d vector): variance of noise.

    Returns:
        (d+1)^(2m) np.array: Estimated moments.

    """
    # Set paramter values
    m = np.size(w_noise, 0)
    N = np.size(w_noise, 1)

    # Set noise vector depending on whether nu is scalar or vector
    if np.size(nu) == 1:
        nu_vec = nu * np.ones(m)
    else:
        nu_vec = np.copy(nu)

    # Initialize matrix which will contain the discrete moments
    M2 = np.zeros(tuple(d + 1 for i in range(2 * m)))

    # Iterative through the possible monomials of the smoothed data
    mi_mat = make_mi_mat(m, d)
    for J_vec in mi_mat:
        for K_vec in mi_mat:
            total_sum = 0

            # Calculate the indice combinations in the summation
            ranges = []
            for K in K_vec:
                ranges.append(list(range(K + 1)))
            combo_list = itertools.product(*ranges)

            # Iterate through each indice combination
            for combo in combo_list:
                k_vec = np.array(combo)
                should_continue = False
                if sum(k_vec) == 0:
                    continue

                # Continue if order is odd since E[epsilon^(odd)]=0
                for k in k_vec:
                    if k % 2 != 0 and k != 0:
                        should_continue = True
                if should_continue:
                    continue

                # Add to the needed adjustment term
                const = 1
                exp_eps = 1
                for i, k in enumerate(k_vec):
                    const *= ncr(K_vec[i], k)
                    exp_eps *= np.prod(range(k - 1, 0,
                                             -2)) * nu_vec[i]**(k / 2)
                mu_prev = M2[
                    tuple([j for j in J_vec] +
                          [K_vec[i] - k_vec[i] for i in range(len(K_vec))])]
                total_sum += const * exp_eps * mu_prev

            # Add the unbiased moment to the moment matrix
            idx = [j for j in J_vec] + [j for j in K_vec]
            prod1 = np.prod(w_tilde**J_vec.reshape(-1, 1), axis=0)
            prod2 = np.prod(w_noise**K_vec.reshape(-1, 1), axis=0)
            M2[tuple(idx)] = 1 / N * np.sum(prod1 * prod2) - total_sum

    return (M2)


def make_M(w_noise, d, nu):
    r"""Calculates the expected discrete moments in $\Theta^T \Theta$.

    Args:
        w_learn (m X N np.array): N noisy measurements of the m states.
        w_tilde (m X N np.array): N smoothed measurements of the m states.
        d (int): maximum degree for the returned moments.
        nu (float or m vector): variance of noise.

    Returns:
        (d+1)^(2m) np.array: Estimated moments.

    """
    # Set paramter values
    m = np.size(w_noise, 0)
    N = np.size(w_noise, 1)

    # Set noise vector depending on whether nu is scalar or vector
    if np.size(nu) == 1:
        nu_vec = nu * np.ones(m)
    else:
        nu_vec = np.copy(nu)

    # Initialize matrix which will contain the discrete moments
    M = np.zeros(tuple(d + 1 for i in range(m)))

    # Iterative through the possible monomials of the smoothed data
    mi_mat = make_mi_mat(m, d)
    for K_vec in mi_mat:
        total_sum = 0

        # Calculate the indice combinations in the summation
        ranges = []
        for K in K_vec:
            ranges.append(list(range(K + 1)))
        combo_list = itertools.product(*ranges)

        # Iterate through each indice combination
        for combo in combo_list:
            k_vec = np.array(combo)
            should_continue = False
            if sum(k_vec) == 0:
                continue

            # Continue if order is odd since E[epsilon^(odd)]=0
            for k in k_vec:
                if k % 2 != 0 and k != 0:
                    should_continue = True
            if should_continue:
                continue

            # Add to the needed adjustment term
            const = 1
            exp_eps = 1
            for i, k in enumerate(k_vec):
                const *= ncr(K_vec[i], k)
                exp_eps *= np.prod(range(k - 1, 0, -2)) * nu_vec[i]**(k / 2)
            mu_prev = M[tuple([K_vec[i] - k_vec[i]
                               for i in range(len(K_vec))])]
            total_sum += const * exp_eps * mu_prev

        # Add the unbiased moment to the moment matrix
        idx = [j for j in K_vec]
        prod = np.prod(w_noise**K_vec.reshape(-1, 1), axis=0)
        M[tuple(idx)] = 1 / N * np.sum(prod) - total_sum

    return (M)


def make_G2_from_moments(M2):
    r"""Make $\tilde{Theta}^T \Theta$ estimator."""
    # Extract parameters
    m = int(len(M2.shape) / 2)
    d = int(np.size(M2, 0)) - 1

    # Generate multi-index matrix
    mi_mat = make_mi_mat(m, d)

    # Initialize G2
    p = np.size(mi_mat, 0)
    G2 = np.zeros((p, p))

    # Add elements to G2 iteratively
    for i, mi1 in enumerate(mi_mat):
        for j, mi2 in enumerate(mi_mat):
            idx = [i for i in mi1] + [i for i in mi2]
            G2[i, j] = M2[tuple(idx)]
    return (G2)


def make_G_from_moments(M):
    """Make Gramian estimator from moments."""
    # Extract parameters
    m = int(len(M.shape))
    d = int(np.size(M, 0) / 2)

    # Generate multi-index matrix
    mi_mat = make_mi_mat(m, d)

    # Initialize G2
    p = np.size(mi_mat, 0)
    G = np.zeros((p, p))

    # Add elements to G2 iteratively
    for i, mi1 in enumerate(mi_mat):
        for j, mi2 in enumerate(mi_mat):
            idx = mi1 + mi2
            G[i, j] = M[tuple(idx)]
    return (G)


def make_aPC(w, N, nu, wlearn=None, p=3, Q=0, x=None, M=None, W=None):
    """Create the library of aPC polynomials up to degree p."""
    # TODO: This function shouldn't go here, and needs to be debugged
    d = np.size(w, 0)
    mi_mat = make_mi_mat(d, p)
    P = np.size(mi_mat, 0)
    if M is None:
        M = make_M(wlearn, p, nu)

    Phi = np.zeros((np.size(w, 1), P))
    Phi[:, 0] = 1
    Phi_sq = None
    if W is None:
        Phi_sq = np.zeros(P)
        Phi_sq[0] = 1
        C = np.zeros((P, P))
        W = np.zeros((P, P))
        W[0, 0] = 1
        for j in np.arange(1, P):
            alpha_j = mi_mat[j, :]
            for k in range(j):
                tot_sum = 0
                for ell in range(k + 1):
                    alpha_sum = mi_mat[ell, :] + alpha_j
                    mu = M[alpha_sum[0], alpha_sum[1]]
                    tot_sum += W[k, ell] * mu
                C[j, k] = 1 / Phi_sq[k] * tot_sum
            Phi_sq[j] = M[2 * alpha_j[0], 2 * alpha_j[1]] - np.sum(
                C[j, :j]**2 * Phi_sq[:j])
            if Phi_sq[j] < 0:
                Phi_sq[j] = 1
            # Set coefficients
            W[j, j] = 1
            for k in range(j):
                coef = -C[j, k]
                coef_prev = W[k, :k + 1]
                W[j, :k + 1] = W[j, :k + 1] + coef * coef_prev
        print(Phi_sq)
        # for ii in range(np.size(Phi_sq)):
        #     if Phi_sq[ii] < 0:
        #         Phi_sq[ii] = 1e-5
        W_norm = np.transpose(np.transpose(W) * Phi_sq**(-1 / 2))
        W = np.copy(W_norm)

    for j in np.arange(1, P):
        for k in range(j + 1):
            mi = mi_mat[k]
            Phi[:, j] += W[j, k] * np.prod((w.T**mi).T, axis=0)
    if Phi_sq is None:
        return Phi, M, W
    else:
        return Phi, M, W, Phi_sq


def get_basis_size(m, d):
    """Get basis size of system with m states and maximum degree d."""
    p = int(sps.factorial(m + d) / (sps.factorial(m) * sps.factorial(d)))
    return (p)


def make_mi_mat(m, d):
    """Given a maximum order and dimension of space, find multi-index matrix.

    :param m: dimension of space (int)
    :param d: maximum polynomial order (int)
    :return multiindex_matrix: P possible multi-indices (P x d np.array)
                               P = (pd!)/(p!d!)
    """
    # Find number of multi-indices
    p = int(sps.factorial(m + d) / (sps.factorial(m) * sps.factorial(d)))

    # Initialize multi-index matrix
    mi_matrix = np.zeros((p, m), dtype=np.int16)

    # For each fixed order, calculate the multi-indices
    row = 0
    for current_order in range(d + 1):
        if current_order == 0:
            row += 1
        else:
            used_rows = int(sps.comb(current_order + m - 1, current_order))
            mi_matrix_p = make_mi_mat_p(m, current_order, used_rows)
            mi_matrix[row:row + used_rows, :] = mi_matrix_p
            row += used_rows
    return (mi_matrix)


def make_mi_mat_p(m, d, rows):
    """Given a FIXED order and dimension of space, find multi-index matrix.

    :param m: dimension of stochastic space (int)
    :param d: fixed polynomial order (int)
    :param rows: number of multi-index vectors to return (int)
    :return multiindex_matrix: rows possible multi-indices (rows x d np.array)
    """
    mi_matrix_p = np.zeros((rows, m), dtype=np.int16)

    # Put all orders in first element of first row
    mi_matrix_p[0, 0] = d

    # If there is a row left continue
    if rows > 1:
        j = 1
        # If the order is greater than zero continue
        while mi_matrix_p[j - 1, 0] > 0:
            # Subtract one from row above
            mi_matrix_p[j, 0] = mi_matrix_p[j - 1, 0] - 1
            # Find new order, number of elements, and rows of next subsystem
            d_new = m - 1
            p_new = d - mi_matrix_p[j, 0]
            used_rows = int(sps.comb(p_new + d_new - 1, p_new))
            # Fill in the first column of matrix with correct order
            mi_matrix_p[j + 1:j + used_rows, 0] = mi_matrix_p[j, 0]
            # Recursively calculate submatrix next to the column filled above
            mi_submatrix_p = make_mi_mat_p(d_new, p_new, used_rows)
            mi_matrix_p[j:j + used_rows, 1:m] = mi_submatrix_p
            j += used_rows
    return mi_matrix_p
