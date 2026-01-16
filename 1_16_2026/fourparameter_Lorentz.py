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
        iter_dir = f'1_16_2026/FourParameter/Lorentz2/iteration_{iter_num:03d}'
        result = u.defectandmirrors2(width = 500, length = 337, mirrorxlength = 218, mirrorlength = 382, 
                                    ylength = x[0], xlength = x[1], widthx = x[2], widthy = x[3], numholes=16,
                                    excitation= mp.Ez, resolution = 32, smoothing = True,
                                    findModes = True, modevisulization = True, showgeo =True, ModeVolume = True, 
                                    dir =iter_dir, lorentz = True)
        
        with open(os.path.join(iter_dir,"optimization.txt"), "w") as f:
                f.write("Optimization Results\n")
                f.write("="*30 + "\n")
                f.write("Optimization Parameters:\n")
                f.write(f"  Iteration: {iter_num} \n")
                f.write(f"  Taper Y Length: {x[0]} \n")
                f.write(f"  Taper X Length: {x[1]} \n")
                f.write(f"  Number of defect holes: {9}\n")
                f.write("="*30 + "\n")

        return result

    def objective(x): 
        result = compute(x)
        return result
    
    return objective 

def run_optimizer(x0: list):
    objective = optimizer()
    res = minimize(
        objective, 
        x0 = x0, 
        method = 'Nelder-Mead',
        bounds=[
             (0,np.inf), 
             (0,np.inf),
             (2.3, np.inf),
             (2.4, np.inf)
        ]
    )

    return res

x0 = [254, 118,2.3,3.6]
run_optimizer(x0)