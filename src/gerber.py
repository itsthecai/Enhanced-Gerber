"""
Name    : gerber.py
Author  : Yinsen Miao
Contact : yinsenm@gmail.com
Time    : 7/1/2021
Desc    : Compute Gerber Statistics
"""
import numpy as np
from numpy import diag, inf
from numpy import copy, dot
from numpy.linalg import norm
import pandas as pd

def is_psd_def(cov_mat):
    """
    :param cov_mat: covariance matrix of p x p
    :return: true if positive semi definite (PSD)
    """
    return np.all(np.linalg.eigvals(cov_mat) > -1e-6)


class ExceededMaxIterationsError(Exception):
    def __init__(self, msg, matrix=[], iteration=[], ds=[]):
        self.msg = msg
        self.matrix = matrix
        self.iteration = iteration
        self.ds = ds

    def __str__(self):
        return repr(self.msg)
    

def nearcorr(A, tol=[], flag=0, max_iterations=100, n_pos_eig=0,
             weights=None, verbose=False,
             except_on_too_many_iterations=True):
    """
    X = nearcorr(A, tol=[], flag=0, max_iterations=100, n_pos_eig=0,
        weights=None, print=0)

    Finds the nearest correlation matrix to the symmetric matrix A.

    ARGUMENTS
    ~~~~~~~~~
    A is a symmetric numpy array or a ExceededMaxIterationsError object

    tol is a convergence tolerance, which defaults to 16*EPS.
    If using flag == 1, tol must be a size 2 tuple, with first component
    the convergence tolerance and second component a tolerance
    for defining "sufficiently positive" eigenvalues.

    flag = 0: solve using full eigendecomposition (EIG).
    flag = 1: treat as "highly non-positive definite A" and solve
    using partial eigendecomposition (EIGS). CURRENTLY NOT IMPLEMENTED

    max_iterations is the maximum number of iterations (default 100,
    but may need to be increased).

    n_pos_eig (optional) is the known number of positive eigenvalues
    of A. CURRENTLY NOT IMPLEMENTED

    weights is an optional vector defining a diagonal weight matrix diag(W).

    verbose = True for display of intermediate output.
    CURRENTLY NOT IMPLEMENTED

    except_on_too_many_iterations = True to raise an exeption when
    number of iterations exceeds max_iterations
    except_on_too_many_iterations = False to silently return the best result
    found after max_iterations number of iterations

    ABOUT
    ~~~~~~
    This is a Python port by Michael Croucher, November 2014
    Thanks to Vedran Sego for many useful comments and suggestions.

    Original MATLAB code by N. J. Higham, 13/6/01, updated 30/1/13.
    Reference:  N. J. Higham, Computing the nearest correlation
    matrix---A problem from finance. IMA J. Numer. Anal.,
    22(3):329-343, 2002.
    """

    # If input is an ExceededMaxIterationsError object this
    # is a restart computation
    if (isinstance(A, ExceededMaxIterationsError)):
        ds = copy(A.ds)
        A = copy(A.matrix)
    else:
        ds = np.zeros(np.shape(A))

    eps = np.spacing(1)
    if not np.all((np.transpose(A) == A)):
        raise ValueError('Input Matrix is not symmetric')
    if not tol:
        tol = eps * np.shape(A)[0] * np.array([1, 1])
    if weights is None:
        weights = np.ones(np.shape(A)[0])
    X = copy(A)
    Y = copy(A)
    rel_diffY = inf
    rel_diffX = inf
    rel_diffXY = inf

    Whalf = np.sqrt(np.outer(weights, weights))

    iteration = 0
    while max(rel_diffX, rel_diffY, rel_diffXY) > tol[0]:
        iteration += 1
        if iteration > max_iterations:
            if except_on_too_many_iterations:
                if max_iterations == 1:
                    message = "No solution found in "\
                              + str(max_iterations) + " iteration"
                else:
                    message = "No solution found in "\
                              + str(max_iterations) + " iterations"
                raise ExceededMaxIterationsError(message, X, iteration, ds)
            else:
                # exceptOnTooManyIterations is false so just silently
                # return the result even though it has not converged
                return X

        Xold = copy(X)
        R = X - ds
        R_wtd = Whalf*R
        if flag == 0:
            X = proj_spd(R_wtd)
        elif flag == 1:
            raise NotImplementedError("Setting 'flag' to 1 is currently\
                                 not implemented.")
        X = X / Whalf
        ds = X - R
        Yold = copy(Y)
        Y = copy(X)
        np.fill_diagonal(Y, 1)
        normY = norm(Y, 'fro')
        rel_diffX = norm(X - Xold, 'fro') / norm(X, 'fro')
        rel_diffY = norm(Y - Yold, 'fro') / normY
        rel_diffXY = norm(Y - X, 'fro') / normY

        X = copy(Y)

    return X


def proj_spd(A):
    # NOTE: the input matrix is assumed to be symmetric
    d, v = np.linalg.eigh(A)
    A = (v * np.maximum(d, 0)).dot(v.T)
    A = (A + A.T) / 2
    return(A)




def gerber_contribution(ri: float = None, rj: float = None) -> float:
    numerator = ((1+abs(ri)) * (1+abs(rj))) ** 0.5
    deno =  1 + (abs(ri) - abs(rj)) ** 2
    return numerator / deno



def gerber_cov_stat0(rets: np.array, threshold: float=0.5) -> tuple:
    """
    compute Gerber covariance Statistics 0, orginal Gerber statistics, not always PSD
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    assert 1 > threshold > 0, "threshold shall between 0 and 1"
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    cov_mat = np.zeros((p, p))  # store covariance matrix
    cor_mat = np.zeros((p, p))  # store correlation matrix

    for i in range(p):
        for j in range(i + 1):
            neg = 0
            pos = 0
            nn = 0
            for k in range(n):
                if ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])):
                    pos += 1
                elif ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])):
                    neg += 1
                    
            # compute Gerber correlation matrix
            cor_mat[i, j] = (pos - neg) / (pos + neg)
            cor_mat[j, i] = cor_mat[i, j]
            cov_mat[i, j] = cor_mat[i, j] * sd_vec[i] * sd_vec[j]
            cov_mat[j, i] = cov_mat[i, j]
    return cov_mat, cor_mat



def gerber_cov_stat1(rets: np.array, threshold: float=0.5) -> tuple:
    """
    compute Gerber covariance Statistics 1
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    assert 1 > threshold > 0, "threshold shall between 0 and 1"
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    cov_mat = np.zeros((p, p))  # store covariance matrix
    cor_mat = np.zeros((p, p))  # store correlation matrix

    for i in range(p):
        for j in range(i + 1):
            neg = 0
            pos = 0
            nn = 0
            for k in range(n):
                if ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])):
                    pos += 1
                elif ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])):
                    neg += 1
                elif abs(rets[k, i]) < threshold * sd_vec[i] and abs(rets[k, j]) < threshold * sd_vec[j]:
                    nn += 1

            # compute Gerber correlation matrix
            cor_mat[i, j] = (pos - neg) / (n - nn)
            cor_mat[j, i] = cor_mat[i, j]
            cov_mat[i, j] = cor_mat[i, j] * sd_vec[i] * sd_vec[j]
            cov_mat[j, i] = cov_mat[i, j]
    return cov_mat, cor_mat


def gerber_cov_stat2(rets: np.array, threshold: float=0.5) -> tuple:
    """
    compute Gerber covariance Statistics 2
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    U = np.copy(rets)
    D = np.copy(rets)

    # update U and D matrix
    for i in range(p):
        U[:, i] = U[:, i] >= sd_vec[i] * threshold
        D[:, i] = D[:, i] <= -sd_vec[i] * threshold

    # update concordant matrix
    N_CONC = U.transpose() @ U + D.transpose() @ D

    # update discordant matrix
    N_DISC = U.transpose() @ D + D.transpose() @ U
    H = N_CONC - N_DISC
    h = np.sqrt(H.diagonal())

    # reshape vector h and sd_vec into matrix
    h = h.reshape((p, 1))
    sd_vec = sd_vec.reshape((p, 1))

    cor_mat = H / (h @ h.transpose())
    cov_mat = cor_mat * (sd_vec @ sd_vec.transpose())
    return cov_mat, cor_mat


# Calculate the Gerber covariance matrix with asset-specific thresholds 



def dynamic_gerber_cov(rets: np.array, threshold: np.array = np.full((9,), 0.5)) -> tuple:    
    """
    compute Gerber covariance Statistic using dynamic threshold and rewards large movements
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    cov_mat = np.zeros((p, p))  # store covariance matrix
    cor_mat = np.zeros((p, p))  # store correlation matrix

    for i in range(p):
        for j in range(i + 1):
            neg = 0
            pos = 0
            nn = 0
            for k in range(n):
                if ((rets[k, i] >= threshold[i] * sd_vec[i]) and (rets[k, j] >= threshold[j] * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold[i] * sd_vec[i]) and (rets[k, j] <= -threshold[j] * sd_vec[j])):
                    pos += gerber_contribution(rets[k, i],rets[k, j])
                elif ((rets[k, i] >= threshold[i] * sd_vec[i]) and (rets[k, j] <= -threshold[j] * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold[i] * sd_vec[i]) and (rets[k, j] >= threshold[j] * sd_vec[j])):
                    neg += gerber_contribution(rets[k, i],rets[k, j])
                elif abs(rets[k, i]) < threshold[i] * sd_vec[i] and abs(rets[k, j]) < threshold[j] * sd_vec[j]:
                    nn += gerber_contribution(rets[k, i],rets[k, j])

            # compute Gerber correlation matrix
            cor_mat[i, j] = (pos - neg) / (n - nn)
            cor_mat[j, i] = cor_mat[i, j]
            cov_mat[i, j] = cor_mat[i, j] * sd_vec[i] * sd_vec[j]
            cov_mat[j, i] = cov_mat[i, j]
    # return the nearest correlation matrix
    return cov_mat, nearcorr(cor_mat)




# This version doesnt reward stronger signal

# def dynamic_gerber_cov(rets: np.array, threshold: np.array = np.full((9,), 0.5)) -> tuple:    
#     """
#     compute Dynamic Gerber covariance Statistics
#     :param rets: assets return matrix of dimension n x p
#     :param threshold: threshold is between 0 and 1 and with dimension n x 1
#     :return: Gerber covariance matrix of p x p
#     """

#     n, p = rets.shape
#     sd_vec = rets.std(axis=0)
#     U = np.copy(rets)
#     D = np.copy(rets)

#     # update U and D matrix based on asset-specific thresholds
#     for i in range(p):
#         U[:, i] = U[:, i] >= sd_vec[i] * threshold[i]
#         D[:, i] = D[:, i] <= -sd_vec[i] * threshold[i]

#     # update concordant matrix
#     N_CONC = U.transpose() @ U + D.transpose() @ D

#     # update discordant matrix
#     N_DISC = U.transpose() @ D + D.transpose() @ U
#     H = N_CONC - N_DISC
#     h = np.sqrt(H.diagonal())

#     # reshape vector h and sd_vec into matrix
#     h = h.reshape((p, 1))
#     sd_vec = sd_vec.reshape((p, 1))

#     cor_mat = H / (h @ h.transpose())
#     cov_mat = cor_mat * (sd_vec @ sd_vec.transpose())
#     return cov_mat, cor_mat



# test gerber_cov_stat1 and gerber_cov_stat2
if __name__ == "__main__":
    bgn_date = "2018-01-01"
    end_date = "2020-01-01"
    nassets = 4
    file_path = "../data/prcs.csv"
    rets_df = pd.read_csv(file_path, parse_dates=['Date'], index_col=["Date"]).pct_change()[bgn_date: end_date].iloc[:, 0: nassets]
    rets = rets_df.values

    neg_rets_df = rets_df[rets_df < 0].fillna(0)
    neg_rets = neg_rets_df.values

    n, p = rets.shape
    print("n = %d, p = %d" % (n, p))
    cov1_mat, cor1_mat = gerber_cov_stat1(rets)
    cov1_mat, cor1_mat
    print('PSD' if is_psd_def(cov1_mat) else 'NPSD')

    cov2_mat, cor2_mat = gerber_cov_stat2(rets)
    cov2_mat, cor2_mat
    print('PSD' if is_psd_def(cov2_mat) else 'NPSD')

    # Test Dynamic Gerber
    cov3_mat, cor3_mat = dynamic_gerber_cov(rets,np.full((9,), 0.5))
    cov3_mat, cor3_mat
    print('PSD' if is_psd_def(cov3_mat) else 'NPSD')
    
    
    n, p = neg_rets.shape
    print("n = %d, p = %d" % (n, p))
    cov1_mat, cor1_mat = gerber_cov_stat1(neg_rets)
    cov1_mat, cor1_mat
    print('PSD' if is_psd_def(cov1_mat) else 'NPSD')

    cov2_mat, cor2_mat = gerber_cov_stat2(neg_rets)
    cov2_mat, cor2_mat
    print('PSD' if is_psd_def(cov2_mat) else 'NPSD')
    
    
    # Test Dynamic Gerber
    cov3_mat, cor3_mat = dynamic_gerber_cov(neg_rets,np.full((9,), 0.5))
    cov3_mat, cor3_mat
    print('PSD' if is_psd_def(cov3_mat) else 'NPSD')