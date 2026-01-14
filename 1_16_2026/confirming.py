import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from IPython.display import Video 
from meep import mpb
from scipy.optimize import minimize
import math
import os
import utils as u



iter_dir = f'1_16_2026/Confirmations/LorentzOptimization14'
taper = u.tapering(382, 254.909, 5)    
result = u.defectandmirrors(width = 500, length = 337, xlength = 218, ylengths = taper, mirrorlength = 382, excitation= mp.Ez, nummirrors= 16,
                                    findModes = True, modevisulization = True, smoothing = True, resolution = 128, showgeo =True, ModeVolume = True, 
                                    dir =iter_dir, lorentz = True)