import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from IPython.display import Video 
from meep import mpb
from scipy.optimize import minimize
import math
import os
import utils as u

taper = u.tapering(380,180,14)
for i in range(1,13): 
    dir = f'Qfactor/Mirror{i}'
    Qsi, freqs, V = u.defectandmirrors(500, length = 337, xlength= 218, ylengths = taper, mirrorlength=382, excitation = mp.Hz, nummirrors=i, findModes=True,
                                     modevisulization=True, smoothing = True, resolution = 64, showgeo = True, ModeVolume = True, dir = dir, freqsolver = False)
