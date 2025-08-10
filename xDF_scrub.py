#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from sorooshafyouni (University of Oxford, 2019)
"""
import numpy as np
import os, sys
from AC_Utils import *
import itertools
import scipy.signal as ss
from scipy.linalg import toeplitz

# def fast_trace(A, B): return(np.einsum('ij,ji->', A, B))

# Input time series as in original xDF, with NAs masking the scrubbed time points
def xDF_scrub(ts, T,
              method      = 'truncate',
              methodparam = 'adaptive',
              verbose     = False,
              TV          = True,
              copy        = True):
    
    if copy: 
        ts = ts.copy()
    
    if np.shape(ts)[1] != T:
        if verbose: print('Second dimension should be T; the matrix was transposed')
        ts = np.transpose(ts)
    
    N  = np.shape(ts)[0]

    scrubbed_frames = np.where(np.any(np.isnan(ts), axis=0))[0]
    t = T - len(scrubbed_frames)

    ts = ts - np.nanmean(ts, axis=1)[:, None]
    norms = np.nanmean(ts**2, axis=1)[:, None]

    # Mask NaNs with 0s for summation
    if t < T: ts[:, scrubbed_frames] = 0

    # Calculate denominator according to acf in R
    retained_frames = np.setdiff1d(np.arange(T), scrubbed_frames)
    list_pairs = list(itertools.combinations(retained_frames, 2))
    n_pairs = [t] + list(np.bincount([y - x for (x, y) in list_pairs], minlength=T)[1:])
    n_denom = n_pairs + np.arange(T)

    # Calculate autocorrelation
    ac = np.concatenate([ss.correlate(ts[i, :], ts[i, :], method='fft')[None, -T:] / n_denom for i in np.arange(N)], axis=0) / norms
    ac = ac[:, 1:]

    # Calculate cross-correlation
    xc = np.zeros((N, N, 2 * T - 1))
    XX = np.triu_indices(N, 1)[0]
    YY = np.triu_indices(N, 1)[1]

    for (i, j) in zip(XX, YY):
    
        xc[i, j, :] = ss.correlate(ts[i, :], ts[j, :], method='fft') / (np.sqrt(norms[i] * norms[j]) * np.concatenate((np.flip(n_denom[1:]), n_denom)))
    
    xc = xc + np.transpose(xc, (1, 0, 2))

    # Extract positive and negative cross-correlations
    xc_p = xc[:, :, :(T - 1)]
    xc_p = np.flip(xc_p, axis=2)
    xc_n = xc[:, :, -(T - 1):] 

    # Extract lag-0 correlations
    rho = np.eye(N) + xc[:, :, T - 1]

    # Regularize!
    if method.lower() == 'tukey':
        if methodparam == '':
            M = np.sqrt(T)
        else: M = methodparam
        if verbose: print('AC regularization: Tukey tapering of M = ' + str(int(np.round(M))))
        ac = tukeytaperme(ac, T - 1, M)
        xc_p = tukeytaperme(xc_p, T - 1, M)
        xc_n = tukeytaperme(xc_n, T - 1, M)
        
    elif method.lower() == 'truncate':

        if type(methodparam) == str: # Adaptive truncation
            if methodparam.lower() != 'adaptive':
                raise ValueError('What?! Choose adaptive as the option or pass an integer for truncation')
            if verbose: print('AC regularization: adaptive truncation')
           
            ac, bp = shrinkme(ac, T - 1)

            for (i, j) in itertools.product(np.arange(N), np.arange(N)):
    
                maxBP = np.max([bp[i], bp[j]])
                xc_p[i, j, :] = curbtaperme(xc_p[i, j, :], T - 1, maxBP, verbose=False)
                xc_n[i, j, :] = curbtaperme(xc_n[i, j, :], T - 1, maxBP, verbose=False)

        elif type(methodparam) == int: # Non-adaptive truncation
            if verbose: print('AC regularization: non-adaptive truncation on M = ' + str(methodparam))         
            ac = curbtaperme(ac, T - 1, methodparam)
            xc_p = curbtaperme(xc_p, T - 1, methodparam)
            xc_n = curbtaperme(xc_n, T - 1, methodparam)
            
        # else: raise ValueError('Method parameter for truncation method should be either str or int')
    
    # Estimate variance (big formula)
    wgt = np.array(n_pairs[1:])

    VarHatRho = (t * (1 - rho**2)**2 \
                    + rho**2 * np.sum(wgt * (ac[:, None, :]**2 + ac[None, :, :]**2 + xc_p**2 + xc_n**2), axis=2) \
                    - 2 * rho * np.sum(wgt * ((ac[:, None, :] + ac[None, :, :]) * (xc_p + xc_n)), axis=2) \
                    + 2 * np.sum(wgt * (ac[:, None, :] * ac[None, :, :] + xc_p * xc_n), axis=2)) / (t**2)

    # for (i, j) in zip(XX, YY):
        
    #     r = rho[i, j]

    #     Sigx = toeplitz(ac[i, :])[np.ix_(retained_frames, retained_frames)]
    #     Sigy = toeplitz(ac[j, :])[np.ix_(retained_frames, retained_frames)]
    
    #     Sigxy = (np.triu(toeplitz(np.insert(xc_p[i, j, :], 0, r)), k=1) + np.tril(toeplitz(np.insert(xc_n[i, j, :], 0, r))))[np.ix_(retained_frames, retained_frames)]

    #     Sigyx = Sigxy.T

    #     VarHatRho[i, j] = ((r**2 / 2) * fast_trace(Sigx, Sigx) + (r**2 / 2) * fast_trace(Sigy, Sigy) \
    #                         + r**2 * fast_trace(Sigyx, Sigxy) + fast_trace(Sigxy, Sigxy) + fast_trace(Sigx, Sigy) \
    #                         - 2 * r * fast_trace(Sigx, Sigxy) - 2 * r * fast_trace(Sigy, Sigyx)) / t**2

    # VarHatRho = VarHatRho + VarHatRho.T
        
    # Truncate to theoretical variance
    if TV: VarHatRho = np.maximum(VarHatRho, (1 - rho**2)**2 / t)

    VarHatRho[range(N), range(N)] = 0
    
    return VarHatRho
    
# Disable verbose
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore verbose
def enablePrint():
    sys.stdout = sys.__stdout__    