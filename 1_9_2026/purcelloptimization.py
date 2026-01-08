import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from IPython.display import Video 
from meep import mpb
from scipy.optimize import minimize
import math
import os
import utils as u

def optimizer():
    iteration_counter = [0]


    def compute(x): 
        x = np.array(x, dtype = float)
        

        iteration_counter[0] += 1
        iter_num = iteration_counter[0]
        iter_dir = f'Optimization2/iteration_{iter_num:03d}'
        taper = u.tapering(382, x[0], x[1])    
        _, __, ___, result = u.defectandmirrors(width = 500, length = 337, xlength = 218, ylengths = taper, mirrorlength = 382, excitation= mp.Hz, nummirrors= 6,
                                    findModes = True, modevisulization = False, smoothing = True, resolution = 32, showgeo =True, ModeVolume = True, 
                                    dir =iter_dir, freqsolver = False, lorentz = False)
        
        with open(os.path.join(iter_dir,"optimization.txt"), "w") as f:
                f.write("Optimization Results\n")
                f.write("="*30 + "\n")
                f.write("Optimization Parameters:\n")
                f.write(f"  Iteration: {iter_num} \n")
                f.write(f"  Taper Length: {x[0]} \n")
                f.write(f"  Number of defect holes: {x[1]}\n")
                f.write("="*30 + "\n")

        return result

    def objective(x): 
        _, __, ___, result = compute(x)
        return -result
    
    return objective 

def run_optimizer(x0: list):
    objective = optimizer()
    res = minimize(
        objective, 
        x0 = x0, 
        method = 'Nelder-Mead',
    )

    return res

x0 = [90,14]
run_optimizer(x0)
