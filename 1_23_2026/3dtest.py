import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from IPython.display import Video 
from meep import mpb
from scipy.optimize import minimize
import math
import os
import utils as u

x = [354.7021776961128   , 145.35750049052075 ,2.3126306430986117,4.05701757038457]
dir = f'1_23_2026/3d'
u.DandMirrors3D(width = 500, length = 337, mirrorxlength = 218, mirrorlength = 382, 
                                    ylength = x[0], xlength = x[1], widthx = x[2], widthy = x[3], numholes=12,
                                    excitation= mp.Ez, resolution = 48, smoothing = True,
                                    findModes = True, modevisulization = False, showgeo =True, ModeVolume = True, 
                                    dir =dir, lorentz = False)
