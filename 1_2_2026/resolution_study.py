import meep as mp
import numpy as np
from collections import defaultdict
import utils as u
import math
import pickle  

rank = mp.my_rank()
is_master = mp.am_master()

taper = u.tapering(380, 180, 10)

resolutions = [128, 256, 512]
num_mirrors = range(1, 17)

Qs = defaultdict(list)
freqs = defaultdict(list)
Vs = defaultdict(list)

for res in resolutions:
    for nm in num_mirrors:

        if is_master:
            print(f"Running resolution={res}, nummirrors={nm}", flush=True)

        Qsi, freqsi, Vi = u.defectandmirrors(
            500,
            length=337,
            xlength=218,
            ylengths=taper,
            mirrorlength=382,
            excitation=mp.Hz,
            nummirrors=nm,
            findModes=True,
            modevisulization=False,
            smoothing=False,
            resolution= res
        )

        if is_master:
            Qs[res].append(Qsi)
            freqs[res].append(freqsi)
            Vs[res].append(Vi)

if is_master:
    output_data = {
        "Qs": dict(Qs),
        "freqs": dict(freqs),
        "Vs": dict(Vs),
        "resolutions": resolutions,
        "num_mirrors": list(num_mirrors),
    }

    with open("meep_results.pkl", "wb") as f:
        pickle.dump(output_data, f)

    print("Saved results to meep_results.pkl", flush=True)
