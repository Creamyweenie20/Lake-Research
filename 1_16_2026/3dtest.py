import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from IPython.display import Video 
from meep import mpb
from scipy.optimize import minimize
import math
import os
import utils as u

x = [359.1355257028107 , 131.47655354850943,2.4521801477615313,3.7409799048829386]
dir = f'confirming/Purcell2/it100'
_,__,result, ___ = u.defectandmirrors2(width = 500, length = 337, mirrorxlength = 218, mirrorlength = 382, 
                                    ylength = x[0], xlength = x[1], widthx = x[2], widthy = x[3], numholes=12,
                                    excitation= mp.Ez, resolution = 128, smoothing = True,
                                    findModes = True, modevisulization = True, showgeo =True, ModeVolume = True, 
                                    dir =dir, lorentz = False)