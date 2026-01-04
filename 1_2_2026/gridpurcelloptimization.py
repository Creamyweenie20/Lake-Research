import meep as mp
import matplotlib.pyplot as plt
import numpy as np
from meep import mpb
from scipy.optimize import minimize
import utils as u

#taper = u.tapering(380, 180, 10)
mirrors = range(4,19)
resolution = 128
num_tapering = range(6,17)

for i in num_tapering:
    for j in mirrors:
        taper = u.tapering(380,180,i)
        dir = f'Optimization_Purcell/numoftapers_{i}/NumMirrors_{j}'
        Qsi, freqs, V = u.defectandmirrors(500, length = 337, xlength= 218, ylengths = taper, mirrorlength=382, excitation = mp.Hz, nummirrors=j, findModes=True,
                                     modevisulization=True, smoothing = True, resolution = resolution, showgeo = True, ModeVolume = True, dir = dir)
