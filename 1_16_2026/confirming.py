import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from IPython.display import Video 
from meep import mpb
from scipy.optimize import minimize
import math
import os
import utils as u



iter_dir = f'1_16_2026/Confirmations/Ofnewtapers'
x0 = [254, 109]
result = u.defectandmirrors(width = 500, length = 337, mirrorxlength = 218, mirrorlength = 382, 
                                    ylength = x0[0], xlength = x0[1], nummirrors=8, numdefects=9,
                                    excitation= mp.Ez, resolution = 16, smoothing = True,
                                    findModes = True, modevisulization = True, showgeo =True, ModeVolume = True, 
                                    dir =iter_dir, lorentz = True)