import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from meep import mpb
from scipy.optimize import minimize
import utils as u

taper = u.tapering(380, 180, 10)
Qsi, freqs = u.defectandmirrors(500, length = 337, xlength= 218, ylengths = taper, mirrorlength=382, excitation = mp.Hz, nummirrors=10, findModes=True,
                                     modevisulization=False, smoothing = False, resolution = 128)