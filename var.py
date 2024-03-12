from collections.abc import Collection
import numpy as np
import math
from scipy.stats import norm
import argparse

def d1(s0, vol, r, k, t):
    return (np.log(s0 / k) + (r + 0.5 * vol ** 2) * t) / (vol * t ** 0.5)

def call_price(s0, vol, r, k, t):
    d1c = d1(s0, vol, r, k, t)
    return s0 * norm.cdf(d1c) - k * math.exp(-r * t) * norm.cdf(d1c - vol * t ** 0.5) 

def put_price(s0, vol, r, k, t):
    d1p = d1(s0, vol, r, k, t)
    return k * math.exp(-r * t) * norm.cdf(-d1p + vol * t ** 0.5) - s0 * norm.cdf(-d1p)

def main(vol, s0, miu, r, k_call, k_put, T, h, n, confidence, portfolio = lambda c, p: p - c):
    p0, c0 = put_price(s0, vol, r, k_put, T), call_price(s0, vol, r, k_call, T)
    v0 = portfolio(c0, p0)

    T -= h

    St = s0 * np.exp((miu - 0.5 * vol ** 2) * h +
        np.random.default_rng().normal(0, 1, n) * vol * (h ** 0.5))

    vt = portfolio(call_price(St, vol, r, k_call, T),
                   put_price(St, vol, r, k_put, T))

    vvar = v0 - vt
    vvar = np.sort(vvar);
    var = np.percentile(vvar, 100 * confidence)
    es = np.mean(vvar[vvar >= var])
    #ivar = round((confidence) * n)
    #var = vvar[ivar]
    #calculate ES
    #es = np.mean(vvar[range(math.floor(confidence * n), n)])
    return vvar, var, es

def display(vvar, bins, var=None, es=None, n=None, path=None):
    try:
        import matplotlib.pyplot as plt
        y, _, _ = plt.hist(vvar, bins=bins, density=True)
        if var != None and es != None and n != None:
            plt.vlines([var, es], ymin=0, ymax=y.max(), color='r') 
        plt.show()
        if path: plt.savefig(path)
    except:
        print("unable to display")
    

if __name__ == "__main__":
#vol = 0.4, portfolio = lambda c, p: p - c,
 #       s0 = 100, miu = 0.05, r = 0.05, k1 = 105, k2 = 95, n = 10000, T = 10, confidence = .95, h = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vol', default=0.4, type=float)
    parser.add_argument('-s', '--spot', default=100, type=float)
    parser.add_argument('-m', '--miu', default=0.1, type=float)
    parser.add_argument('-r', '--rate', default=0.05, type=float)
    parser.add_argument('-kc', '--k_call', default=105, type=float)
    parser.add_argument('-kp', '--k_put', default=95, type=float)
    parser.add_argument('-t', '--time_to_expiry', default=10, type=float)
    parser.add_argument('-hor', '--horizon', default=1, type=float)
    parser.add_argument('-c', '--confidence', default=.95, type=float)
    parser.add_argument('-n', '--num_of_trials', default=10000, type=int)
    parser.add_argument('-f', '--save_path', type=str)
    parser.add_argument('-b', '--display_bins', default=100, type=int)
    args = parser.parse_args()
    print(args)
    vvar, var, es = main(args.vol,
                         args.spot,
                         args.miu,
                         args.rate,
                         args.k_call,
                         args.k_put,
                         args.time_to_expiry,
                         args.horizon,
                         args.num_of_trials,
                         args.confidence
                         )

    print(f"VaR: {var} ES: {es}")
    display(vvar, args.display_bins, var, es, args.num_of_trials, args.save_path)
