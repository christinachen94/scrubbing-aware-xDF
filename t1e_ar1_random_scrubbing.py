from xDF_scrub import *
from xDF import *
import os
import pandas as pd
import argparse
import random
import pathos.multiprocessing as mp
import statsmodels.api as sm

parser = argparse.ArgumentParser()
parser.add_argument('--retained_length', help='number of retained timepoints post scrubbing', type=int)
parser.add_argument('--ac', help='autocorrelation', type=float)
parser.add_argument('--output', help='output directory', type=str)
args = parser.parse_args()

t = args.retained_length
ac = args.ac
output = args.output

def generate_ar1(i, ac, T, t):

	ts = sm.tsa.arma_generate_sample(np.array([1, -ac]), [1], nsample=(2, T), axis=1, burnin=2000)

	removed_frames = random.sample(range(T), T - t)

	ts[:, removed_frames] = np.nan

	masked_var = xDF_scrub(ts, T, method='truncate')[0, 1]
	scrubbed_ts = np.delete(ts, removed_frames, axis=1)
	scrubbed_rho = np.corrcoef(scrubbed_ts)[0, 1]
	scrubbed_var = xDF_Calc(scrubbed_ts, T - len(removed_frames), method='truncate')['v'][0, 1]

	return(np.array([masked_var, scrubbed_var, scrubbed_rho]))
        
with mp.Pool() as pool:

    results = np.vstack(pool.map(lambda i: generate_ar2(i, c1, c2, 2000, t), range(1000000)))

np.savetxt(f'{output}/ar1_{ac}_retained_{t}.txt', results)
