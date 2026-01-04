import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from meep import mpb
from scipy.optimize import minimize
import utils as u

taper = u.tapering(380, 180, 10)
#Qsi, freqs, V = u.defectandmirrors(500, length = 337, xlength= 218, ylengths = taper, mirrorlength=382, excitation = mp.Hz, nummirrors=10, findModes=True,
#                                    modevisulization=False, smoothing = False, resolution = 256, showgeo = False, ModeVolume = True)
mirrors = range(1,13)
resolution = [256] 

for i in resolution: 
    for j in mirrors: 
        dir = f'Results2/Resolution_{i}/NumMirrors_{j}'
        Qsi, freqs, V = u.defectandmirrors(500, length = 337, xlength= 218, ylengths = taper, mirrorlength=382, excitation = mp.Hz, nummirrors=j, findModes=True,
                                     modevisulization=False, smoothing = False, resolution = i, showgeo = False, ModeVolume = True, dir = dir)
