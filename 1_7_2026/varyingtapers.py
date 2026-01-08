import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from IPython.display import Video 
from meep import mpb
from scipy.optimize import minimize
import math
import os
import utils as u

mirrors = 12
resolution = 64
num_tapering = range(2,17)

for i in num_tapering:
        taper = u.tapering(380,180,i)
        dir = f'Taper_effects/numoftapers_{i}'
        Qsi, freqs, V = u.defectandmirrors(500, length = 337, xlength= 218, ylengths = taper, mirrorlength=382, excitation = mp.Hz, nummirrors=mirrors, findModes=True,
                                     modevisulization=True, smoothing = True, resolution = resolution, showgeo = True, ModeVolume = True, dir = dir)