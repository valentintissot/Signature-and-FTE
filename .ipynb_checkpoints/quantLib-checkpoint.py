### Short quantitative finance Library
import numpy as np
from numpy import random as rdm
from scipy.stats import norm as nm

def BM(Ts,J,seed = None):
    """Simulate J Brownian paths on the time grid Ts (column vector)."""
    if seed is not None: rdm.seed(seed)
    # Time increments
    dts = np.diff(Ts,axis = 0)
    # Brownian increments
    dW  = np.sqrt(dts) * rdm.randn(len(dts),J)
    return np.vstack((np.zeros((1,J)),np.cumsum(dW,axis = 0)))


def bsVanilla(x0,K,r,delta,sig,T):
    """Black-Scholes price and greeks for vanilla options (call and put)"""
    # Discount factor
    D = np.exp(-r*T)
    # Forward Price
    F = x0/D * np.exp(-delta*T)
    # Volatility over [0,T]
    sigT = sig * np.sqrt(T)
    # Formula when volatility and time to maturity are positive
    if sigT > 0:
        # Thresholds (i = 1,2)
        d = lambda i: np.log(F/K)/sigT - (-1)**i * sigT/2
        # Call Price
        C = D * (F * nm.cdf(d(1)) - K * nm.cdf(d(2)))
        # Delta of Call Option
        dC = np.exp(-delta*T) * nm.cdf(d(1))
        # Gamma of Call/Put Option
        g = np.exp(-delta*T) * nm.pdf(d(1)) / (x0 * sigT)
        # Vega of Call/Put Option
        v = g * x0**2 * sig * T
    elif T == 0: 
        C, dC, g, v = np.maximum(x0 - K,0.), 1 * (x0 >= K), 0.*x0, 0.*x0
    elif sig == 0: 
        C, dC, g, v = D * np.maximum(F - K,0.), np.exp(-delta*T) * (F >= K), 0.*x0, 0.*x0
    # Put Price and Delta (use Put-Call parity!)
    P, dP = C + D * (K - F), dC - 1
    return  C, P , dC, dP, g, v 
