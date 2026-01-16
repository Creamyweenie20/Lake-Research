import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from IPython.display import Video 
from meep import mpb
from scipy.optimize import minimize
import math
import os
import utils as u
iter_dir =f'1_16_2026/TestingFourParameters'
x = [322, 122]
_,__,___, result = u.defectandmirrors2(width = 500, length = 337, mirrorxlength = 218, mirrorlength = 382, 
                                    ylength = x[0], xlength = x[1], widthx = 1, widthy = 1, numholes=12,
                                    excitation= mp.Ez, resolution = 16, smoothing = True,
                                    findModes = True, modevisulization = True, showgeo =True, ModeVolume = True, 
                                    dir =iter_dir, lorentz = False)