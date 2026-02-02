import numpy as np
from scipy import optimize 
import matplotlib.pyplot as plt
from Newton import Newton


c = 3e8
w = 2*np.pi*193.824246e12
k0 = w/c
d = 220e-9
n1 = 3.4
n2 = 1.0003

def f(beta, w, d, n1=3.4, n2=1.0003):
    k0 = w / c
    h = np.sqrt(n1**2 * k0**2 - beta**2)
    alpha = np.sqrt(beta**2 - n2**2 * k0**2)
    return np.tan(h*d/2) - alpha/h

def fprime(beta, w, d, n1=3.4, n2=1.0003):
    k0 = w / c

    if beta <= n2*k0 or beta >= n1*k0:
        return np.nan

    h = np.sqrt(n1**2 * k0**2 - beta**2)
    alpha = np.sqrt(beta**2 - n2**2 * k0**2)

    term1 = -(beta * d) / (2 * h) * (1 / np.cos(h*d/2))**2
    term2 = -beta / (alpha * h)
    term3 = -alpha * beta / h**3

    return term1 + term2 + term3


def f2(beta, w, d, n1=3.4, n2=1.0003):
    k0 = w / c
    h = np.sqrt(n1**2 * k0**2 - beta**2)
    alpha = np.sqrt(beta**2 - n2**2 * k0**2)
    return np.tan(h*d/2) - (n1/n2)**2 * alpha/h

def f2prime(beta, w, d, n1 = 3.4, n2 = 1.0003): 
    k0 = w / c

    if beta <= n2*k0 or beta >= n1*k0:
        return np.nan
    
    h = np.sqrt(n1**2 * k0**2 - beta**2)
    alpha = np.sqrt(beta**2 - n2**2 * k0**2)

    term1 = -(beta * d) / (2 * h) * (1 / np.cos(h*d/2))**2
    term2 = -(n1/n2)**2 * beta / (alpha * h)
    term3 = -(n1/n2)**2 * alpha * beta / h**3

    return term1 + term2 + term3

solver = Newton(f = lambda b: f(b,w,d),
                fprime=lambda b: fprime(b,w,d), 
                bounds = [n2*k0, n1*k0])


root = solver.solve(guess = 1.02 * n2 * k0)
print(f'The propagation constant of the TE fundemental mode is: {root}, which is an effective index of {root/(w/c)}')

solver2 = Newton(f = lambda b: f2(b,w,d),
                fprime=lambda b: f2prime(b,w,d), 
                bounds = [n2*k0, n1*k0])


root2 = solver2.solve(guess = 1.02 * n2 * k0)
print(f'The propagation constant of the TM fundemental mode is: {root2}, which is an effective index of {root2/(w/c)}')

#V = 0.958629 * 2* np.sin()