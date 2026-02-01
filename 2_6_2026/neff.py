import numpy as np
from scipy import optimize 
import matplotlib.pyplot as plt


c = 3e8
w = 2*np.pi*193.824246e12
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


beta = optimize.brentq(
    lambda b: f(b, w, d), 
    1.0001* n2 * w / c, 
    0.99999* n1* w / c
)


print(beta)


def newton_method(guess, w, d, tol=1e-8, iterations=300):
    k0 = w / c
    beta_min = 1.001 * n2 * k0
    beta_max = 0.999 * n1 * k0

    beta = guess

    for i in range(iterations):
        if beta <= beta_min or beta >= beta_max:
            raise ValueError("Initial guess outside guided region")

        f0 = f(beta, w, d)
        fp = fprime(beta, w, d)


        step = -f0 / fp
        new = beta + 0.3 * step   

        if new <= beta_min or new >= beta_max:
            new = 0.5 * (beta + beta_min) if step < 0 else 0.5 * (beta + beta_max)

        if abs(new - beta) < tol:
            print(f'The number of iterations were {i+1}')
            return new

        beta = new

k0 = w / c
guess = 1.02 * n2 * k0

#print(guess, n2*w/c)

Beta = newton_method(guess, w, d)
print(Beta)