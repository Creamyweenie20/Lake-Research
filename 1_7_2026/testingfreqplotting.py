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
Qsi, freqs, V = u.defectandmirrors(500, length = 337, xlength= 218, ylengths = taper, mirrorlength=382, excitation = mp.Hz, nummirrors=12, findModes=True,
                                     modevisulization=True, smoothing = True, resolution = 64, showgeo = True, ModeVolume = True, dir = 'plotting', freqsolver = False)
