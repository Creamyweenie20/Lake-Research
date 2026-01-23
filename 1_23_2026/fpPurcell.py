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
        iter_dir = f'1_23_2026/Purcell_FromModeVolume1/iteration_{iter_num:03d}'
        _,__,___, result = u.defectandmirrors2(width = 500, length = 337, mirrorxlength = 218, mirrorlength = 382, 
                                    ylength = x[0], xlength = x[1], widthx = x[2], widthy = x[3], numholes=12,
                                    excitation= mp.Ez, resolution = 32, smoothing = True,
                                    findModes = True, modevisulization = True, showgeo =True, ModeVolume = True, 
                                    dir =iter_dir, lorentz = False)
        
        with open(os.path.join(iter_dir,"optimization.txt"), "w") as f:
                f.write("Optimization Results\n")
                f.write("="*30 + "\n")
                f.write("Optimization Parameters:\n")
                f.write(f"  Iteration: {iter_num} \n")
                f.write(f"  Taper Y Length: {x[0]} \n")
                f.write(f"  Taper X Length: {x[1]} \n")
                f.write(f"  Width X: {x[2]}\n")
                f.write(f'  Width Y: {x[3]}\n')
                f.write("="*30 + "\n")

        return result

    def objective(x): 
        result = compute(x)
        return -result
    
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
             (1, np.inf),
             (1, np.inf)
        ]
    )

    return res

x0 = [348.2811805842387 , 155.9029487102001,2.3124840164108074,3.783910803678695]
run_optimizer(x0)