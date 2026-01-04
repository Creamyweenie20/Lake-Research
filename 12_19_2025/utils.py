"""
This Script contains all functions utilized for constructing my first Photonic Crystal

This includes optimization and general geometry constructions, the necessary libraries to import are: 
numpy
meep
mpb (in the meep library)
matplotlib
scipy.optimize

You will need to be in the respective (conda) environment to import the functions if the libraries are not on your devie 
"""

import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from IPython.display import Video 
from meep import mpb
from scipy.optimize import minimize
import math


def tapering(start: float, end: float, n:int, mode:str = "quad"):
    """ 
    Function that takes in the starting length and tapers to the ending length, with n total increments
    Modes of tapering: (Defaults to Quad)
    1. Linear (Lin)
    2. Quadratic (Quad)
    3. Cubic (Cub)
    4. Exponential (Exp)

    start: starting length

    end: ending length 

    n: integer for specifying number of increments (for now restricted to being even)
    """

    N = np.arange(1,n+1)
    try: 
        match mode.lower():
            case "quad":
                tapered = start - (start-end)*(N/n)**2
            case "lin":
                tapered = start - (start - end) * (N/n)
            case "cub":
                tapered = start - (start-end)* (N/n)**3
            case "exp": 
                c = np.log(start/end)
                tapered = start * np.exp(-c * (N/n))
            case _:
                raise ValueError
            
    except: 
        raise ValueError("Invalid Mode of Tapering: Choose between 'quad', 'lin', 'cub', or 'exp'")
        
    return tapered

def defectandmirrors(width: float, length:float, xlength: float, ylengths: list, mirrorlength: float, nummirrors: int = 8, excitation = mp.Hz,
                     showgeo: bool = True, modevisulization: bool = True, findModes: bool = True, input_freq: float | None = None, 
                     MaxQ_freq: bool = False, resolution: int  = 64, smoothing: bool = True, ModeVolume:bool = True):
    """
    Docstring for defectandmirrors
    
    :param width: Width of Unit Cell in Nanometers
    :param length: Length of Unit Cell in Nanometers
    :param xlength: Diameter of holes in x-direction in Nanometers
    :param ylengths: Array of tapered diameter of holes in defect region, y-direction, in Nanometers
    :param mirrorlength: Major axis length of Mirror Unit Cell in nanometers
    :param nummirrors: Number of mirrors in the Mirror Section on one side
    :param excitation: Meep Excitation Signal used in simulations, defaulted to Z-component magnetic field
    :param showgeo: Show Geometry of the simulation before any excitation, defaults to True
    :param modevisulization: Visualize the Modes in the Photonic Crystal, default set to True
    :param findModes: Use Harminv to find the modes of the photonic crystal with the specified excitation signal
    :param input_freq: Input Frequency in Meep normalized units, do not have MaxQ_freq if you do want a specified frequency
    :param MaxQ_freq: Set input_freq variable to the highest Q frequency found from HarmInv, need findModes to be True
    :param resolution: Set resolution of the computational space, defaults to 64
    :param smoothing: Subpixel smoothing enabler, defaults to True
    :param ModeVolume: Calculate the Mode volume with modal_volume_in_box
    """


    Magnetic_field = {mp.Hx, mp.Hy, mp.Hz}

    c = 3e8 # Speed of light for conversion later
    conv = (c/(length*1e3)) 
    mp.verbosity(3)

    # Parameters for the computational cell: 
    pml_padding = length
    padding_to_pml = 4*length
    sx = 2*(len(ylengths))-1 + 2* nummirrors  # We will scale the dimensions of everything else by the length (for example we scaled the paddings above by the length)
    sy = 2*(pml_padding + padding_to_pml + width)/length  

    cell = mp.Vector3(sx,sy)

    # For the silicon we want to use and our approximate target wavelength is below: 
    eps_silicon = 12
    thickness = 220
    wavelength = 1540 

    # Defining our waveguide given the parameters, meep assumes periodic structure: 
    beam = mp.Block(size = (sx, width/length, thickness/length), material = mp.Medium(epsilon = eps_silicon) )
    holes = []
    for i in range(len(ylengths)): 
        holes.append(mp.Ellipsoid(center = mp.Vector3((-sx/2 + .5 + nummirrors) + i, 0, 0), size = mp.Vector3(xlength, ylengths[i], thickness)/length, e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1),
                          material = mp.Medium(epsilon = 1)))
        holes.append(mp.Ellipsoid(center = mp.Vector3((+sx/2 - .5 - nummirrors) - i, 0, 0), size = mp.Vector3(xlength, ylengths[i], thickness)/length, e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1),
                          material = mp.Medium(epsilon = 1)))

    for i in range(nummirrors): 
        holes.append(mp.Ellipsoid( center = mp.Vector3((-sx/2 + 0.5) + i , 0, 0 ), size = mp.Vector3(xlength, mirrorlength, thickness)/length, e1 = mp.Vector3(1,0,0), 
                     e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
        holes.append(mp.Ellipsoid( center = mp.Vector3((+sx/2 - 0.5) - i , 0, 0 ), size = mp.Vector3(xlength, mirrorlength, thickness)/length, e1 = mp.Vector3(1,0,0), 
                     e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))

    geometry = [beam]+holes

    # Other things for simulation: It is to note that say we want to run for a certian number of periods, then we would divide our time by the frequency below
    fcenter = length/wavelength           # In meep units, the frequency is specified by 1/lambda according to https://meep.readthedocs.io/en/latest/Introduction/#units-in-meep
    df = .3

    # Broad source to excite modes
    source = mp.Source(mp.GaussianSource(fcenter, fwidth = df), excitation, mp.Vector3(0, 0))
    pml_layers = [mp.PML(pml_padding/length, direction = mp.Y)]

    if excitation in Magnetic_field:
        sym = [mp.Mirror(mp.X, phase =-1), mp.Mirror(mp.Y, phase = -1)]
    else: 
        sym = [mp.Mirror(mp.X, phase = 1), mp.Mirror(mp.Y, phase = 1)]

    simulation = mp.Simulation(
        cell_size = cell, 
        boundary_layers = pml_layers, 
        eps_averaging=smoothing, 
        geometry = geometry, 
        sources = [source], 
        resolution = resolution,
        symmetries = sym
        )
    
    if showgeo and mp.am_master(): 
        f = plt.figure()
        simulation.plot2D(ax = f.gca())
        f.gca().xaxis.set_visible(False)
        f.suptitle('Geometry of Simulation')
        plt.axvline(x = -(sx/2) + nummirrors , color = 'k', linestyle = '--')
        plt.axvline(x = (sx/2) - nummirrors , color = 'k', linestyle = '--')
        plt.show()
        plt.close()

    freqs = []
    Qs = []
    V_mode = 0
    if findModes: 
        h = mp.Harminv(excitation, mp.Vector3(), fcenter, df)

        simulation.run(mp.at_beginning(mp.output_epsilon),
            mp.after_sources(h),
            until_after_sources=1200)
        
        freqs = [m.freq for m in h.modes]
        Qs = [m.Q for m in h.modes]

        if ModeVolume: 
            V_mode = simulation.modal_volume_in_box(mp.Volume(center = mp.Vector3(), size = mp.Vector3(2*len(ylengths)-1, width/length)))



        
        maxq = max(Qs)
        ind = Qs.index(maxq)
        freq_maxq = freqs[ind]*conv
        if MaxQ_freq:
            input_freq = freq_maxq/conv
        print(f'The Frequency of the highest Q is {freq_maxq} THz')


    if modevisulization and input_freq != None: 
        source_new = mp.Source(mp.GaussianSource(input_freq, fwidth = .1), excitation, mp.Vector3(0, 0))
        sim2 = mp.Simulation(
            cell_size = cell, 
            boundary_layers = pml_layers, 
            geometry = geometry, 
            sources = [source_new], 
            resolution = resolution,
            symmetries = sym
            )
        sim2.run(until_after_sources=600)
        if mp.am_master():
            f = plt.figure(dpi=150)
            plt.axvline(x = -(sx/2) + nummirrors , color = 'k', linestyle = '--')
            plt.axvline(x = (sx/2) - nummirrors , color = 'k', linestyle = '--')
            sim2.plot2D(ax=f.gca(), fields=mp.Ex)
            plt.show()

            f2 = plt.figure(dpi=150)
            plt.axvline(x = -(sx/2) + nummirrors , color = 'k', linestyle = '--')
            plt.axvline(x = (sx/2) - nummirrors , color = 'k', linestyle = '--')
            sim2.plot2D(ax=f2.gca(), fields=mp.Ey)
            plt.show()

            f3 = plt.figure(dpi=150)
            plt.axvline(x = -(sx/2) + nummirrors , color = 'k', linestyle = '--')
            plt.axvline(x = (sx/2) - nummirrors , color = 'k', linestyle = '--')
            sim2.plot2D(ax=f3.gca(), fields=mp.Ez)
            plt.show()

            f4 = plt.figure(dpi=150)
            plt.axvline(x = -(sx/2) + nummirrors , color = 'k', linestyle = '--')
            plt.axvline(x = (sx/2) - nummirrors , color = 'k', linestyle = '--')
            sim2.plot2D(ax=f4.gca(), fields=mp.Hx)
            plt.show()

            f5 = plt.figure(dpi=150)
            plt.axvline(x = -(sx/2) + nummirrors , color = 'k', linestyle = '--')
            plt.axvline(x = (sx/2) - nummirrors , color = 'k', linestyle = '--')
            sim2.plot2D(ax=f5.gca(), fields=mp.Hy)
            plt.show()

            f6 = plt.figure(dpi=150)
            plt.axvline(x = -(sx/2) + nummirrors , color = 'k', linestyle = '--')
            plt.axvline(x = (sx/2) - nummirrors , color = 'k', linestyle = '--')
            sim2.plot2D(ax=f6.gca(), fields=mp.Hz)
            plt.show()

    return Qs, freqs, V_mode

def banddiagram_wmpb(width: float, length: float, xlength: float, ylength: float, show_geometry: bool = True, band_diagram: str = "max", 
                     num_bands: float = 2, int_points: float = 30, resolution: int = 64):
    """
    Docstring for banddiagram_wmpb
    
    :param width: Width of Unit Cell in Nanometers
    :param length: Length of Unit Cell in Nanometers
    :param xlength: Diameter of Holes in X direction
    :param ylength: Diameter of Hole in the Y direction
    :param show_geometry: Show the Geometry of the Unit Cell, Defaults to True meaning yes show
    :param band_diagram: String in which can draw all band gaps or just the maximum band gap, defaults to the max setting
    :param num_bands: Number of Bands MPB calculates, defaults to 2
    :param int_points: interpolation points for k vector, set to 30 by default, makes 32 points for k vector by construction
    :param resolution: Resolution of geometry visualization, set to 64 by default
    """
    
    c = 3e8 
    convfactor = (c/(length*1e3))

    # Parameters for the computational cell: 
    pml_padding = length  
    padding_to_pml = 4*length
    sx = 1       # We will scale the dimensions of everything else by the length (for example we scaled the paddings above by the length)
    sy = 2*(pml_padding + padding_to_pml + width)/length  
    

    cell = mp.Vector3(sx,sy)
    lattice = mp.Lattice(mp.Vector3(1,1))

    # For the silicon we want to use and our approximate target wavelength is below: 
    eps_silicon = 12
    thickness = 220
    wavelength = 1540 

    # Defining our waveguide given the parameters, meep assumes periodic structure: 
    beam = mp.Block(size = (1, width/length, thickness/length), material = mp.Medium(epsilon = eps_silicon) )
    holes = mp.Ellipsoid(size = mp.Vector3(xlength, ylength, thickness)/length, e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1),
                          material = mp.Medium(epsilon = 1))

    geometry = [beam,holes]

    if show_geometry: 
        # Other things for simulation: It is to note that say we want to run for a certian number of periods, then we would divide our time by the frequency below
        fcenter = length/wavelength           # In meep units, the frequency is specified by 1/lambda according to https://meep.readthedocs.io/en/latest/Introduction/#units-in-meep
        df = .5*fcenter

        sym = [mp.Mirror(direction=mp.Y, phase=-1), mp.Mirror(direction = mp.X, phase = -1)]
        # Broad source to excite modes
        source = mp.Source(mp.GaussianSource(fcenter, fwidth = df), mp.Hz, mp.Vector3(0, 0))
        pml_layers = [mp.PML(pml_padding/length, direction =mp.Y)]

        simulation = mp.Simulation(
            cell_size = cell, 
            boundary_layers = pml_layers, 
            geometry = geometry, 
            sources = [source], 
            resolution = resolution, 
            symmetries=sym
        )
        if mp.am_master():
            f = plt.figure(dpi=200)
            simulation.plot2D(ax = f.gca())
            f.gca().xaxis.set_visible(False)
            f.suptitle('Geometry of Simulation')
            plt.show()

    # It was more or less the same to the prior implementation of this code, now we will utlize MPB to get the band diagrams: 
    k_points = [mp.Vector3(0), mp.Vector3(.5)]       # Two dimensional simulation but only getting the x component of the wave vectors
    k_points = mp.interpolate(int_points, k_points)         # Interpolation to get more k vectors for our simulation, set to another power of two
    
    # mpb implementation of the mode solver: 
    ms = mpb.ModeSolver(
        num_bands = num_bands, 
        k_points= k_points, 
        geometry= geometry, 
        geometry_lattice= lattice, 
        resolution= resolution
    )

    # We will split the modes up into TE and TM modes: 
    ms.run_te()
    te = ms.all_freqs       
    te_gaps = ms.gap_list

    ms.run_tm()
    tm = ms.all_freqs
    tm_gaps = ms.gap_list

    plot_k = np.array(k_points)
    plot_te_freqs = np.array(te)*convfactor
    plot_tm_freqs = (np.array(tm))*convfactor

    fig = plt.figure(num = 1,dpi = 200, figsize = (10,10))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(plot_k[:,0].T, plot_te_freqs)

    # Plotting the band gaps: 
    max_te = 0
    max_te_percent = 0 
    half_band_te = 0 
    for gap in te_gaps: 
        if gap[0]>1: 
            [max_te, max_te_percent] = [max(max_te, (gap[2] - gap[1])*convfactor), max(max_te_percent, gap[0])]
            half_band_te = ((gap[2] + gap[1])/2)*convfactor
            mid_to_top_te = gap[2]*convfactor - half_band_te
            if band_diagram == 'all':
                ax1.fill_between(plot_k[:,0], gap[1]*convfactor, gap[2]*convfactor, color = 'blue', alpha = .2)
 
    if band_diagram == 'max':
        indexed = te_gaps[:][0].index(max_te_percent)
        ax1.fill_between(plot_k[:,0], te_gaps[indexed][1]*convfactor, te_gaps[indexed][2]*convfactor, color = 'blue', alpha =.2)

    print(f'The mid-point for the TE mode is at {half_band_te} (dashed black line)')
    ax1.axhline(y = half_band_te, color='k', linestyle='--')

    print(f'The largest bandgap is {max_te} THz thick for TE modes')
    ax1.set_xlabel('$k_x$  $[\\frac{2\pi}{a}]$')
    ax1.set_ylabel('f [THz]')
    ax1.set_title('Band Diagram for TE Modes')
    ax1.set_xlim([0,.5])
    
    ax2 = plt.subplot(2,1,2)
    ax2.plot(plot_k[:,0].T, plot_tm_freqs)
    

    # Plotting the band gaps: 
    max_tm = 0
    max_tm_percent = 0 
    half_band_tm = 0
    for gap in tm_gaps: 
        if gap[0]>1: 
            [max_tm, max_tm_percent] = [max(max_tm, (gap[2] - gap[1])*convfactor), max(max_tm_percent, gap[0])]
            half_band_tm = ((gap[2] + gap[1])/2)*convfactor
            mid_to_top_tm = gap[2]*convfactor - half_band_tm
            if band_diagram == 'all':
                ax2.fill_between(plot_k[:,0], gap[1]*convfactor, gap[2]*convfactor, color = 'red', alpha = .2)
    print(f'The mid-point for the TM mode is at {half_band_tm} (dashed black line)')
    ax2.axhline(y = half_band_tm, color='k', linestyle='--')
    if band_diagram == 'max':
        indexed = tm_gaps[:][0].index(max_tm_percent)
        ax2.fill_between(plot_k[:,0], tm_gaps[indexed][1]*convfactor, tm_gaps[indexed][2]*convfactor, color = 'red', alpha =.2)

    


    print(f'The largest bandgap is {max_tm} THz thick for TM modes' )
    ax2.set_xlabel('$k_x$  $[\\frac{2\pi}{a}]$')
    ax2.set_ylabel('f [THz]')
    ax2.set_title('Band Diagram for TM Modes')
    ax2.set_xlim([0,.5])

    plt.tight_layout()
    plt.show()

    return [mid_to_top_te, half_band_te, mid_to_top_tm, half_band_tm]
    

def optimizer(width: float, difference: float, tolerance: float, target: float, tol_target: float):
    """
    :param width: Width of our unit cell held constant in optimization, in nanometers
    :param difference: Difference between hole axes and unit cell, in nanometers, we hope to constrain with
    :param tolerance: Tolerance to how seperated the TE and TM band gaps are, units of THz
    :param target: Target Frequency in THz
    :param tol_target: Tolerance we allow for the band gap center to be from the target frequency
    """
    cache = {"x": None, "result": None}

    def compute(x):
        x = np.array(x, dtype=float)
        if cache["x"] is not None and np.allclose(x, cache["x"], rtol=1e-12, atol=1e-12):
            return cache["result"]

        result = banddiagram_wmpb(
            width, x[0], x[1], x[2],
            show_geometry=False,
            band_diagram='all'
        )

        cache["x"] = x
        cache["result"] = np.array(result, dtype=float)

        return cache["result"]
    
    def objective(x):
        result = compute(x)
        #v1 = difference - (x[0] - x[1])
        #v2 = difference -(width - x[2])
        #v3 = tolerance - (result[1] - result[3]) 
        #v4 = tol_target - (result[1] - target)
        #v5 = tol_target - (result[3] - target)

        return -result[0] #+ 10*max(0,v1) + 10*max(0,v2) + 5*max(0,v3) + 100*max(0,v4) + 100*max(0,v5)

    def constraint_difference(x):
        length, xlength, ylength = x
        return [
            length - xlength - difference,
            width - ylength - difference
        ]

    def constraint_equality(x):
        result = compute(x)
        return tolerance**2 - (result[1] - result[3])**2 

    def constraint_band(x):
        result = compute(x)
        return [tol_target**2 - (result[1] - target)**2, tol_target**2 - abs(result[3] - target)**2]

    constraints = [
        {'type': 'ineq', 'fun': constraint_difference},
        {'type': 'ineq', 'fun': constraint_equality},
        {'type': 'ineq', 'fun': constraint_band}
    ]

    return objective, constraints

def run_optimizer(x0: list, width: float, difference: float, tolerance: float, target: float, tol_target: float):
    """
    :param x0: Initial guess, [length, xlength, ylength]
    :param width: Width of our unit cell held constant in optimization, in nanometers
    :param difference: Difference between hole axes and unit cell, in nanometers, we hope to constrain with
    :param tolerance: Tolerance to how seperated the TE and TM band gaps are, units of THz
    :param target: Target Frequency in THz
    :param tol_target: Tolerance we allow for the band gap center to be from the target frequency
    """

    objective, constraints = optimizer(width, difference, tolerance, target, tol_target)
    res = minimize(
        objective,
        x0=x0,
        constraints=constraints,
        method = 'COBYLA',
        options={'maxiter': 200, 'tol': 1e-4, 'rhobeg': 20}
    )

def ModeVolumeMirrors(width: float, length:float, xlength: float, ylengths: list, mirrorlength: float, nummirrors: int = 8, excitation = mp.Hz,
                     showgeo: bool = True, resolution: int  = 64, smoothing: bool = True, Bulk: bool = False):
    """
    Docstring for defectandmirrors
    
    :param width: Width of Unit Cell in Nanometers
    :param length: Length of Unit Cell in Nanometers
    :param xlength: Diameter of holes in x-direction in Nanometers
    :param ylengths: Array of tapered diameter of holes in defect region, y-direction, in Nanometers
    :param mirrorlength: Major axis length of Mirror Unit Cell in nanometers
    :param nummirrors: Number of mirrors in the Mirror Section on one side
    :param excitation: Meep Excitation Signal used in simulations, defaulted to Z-component magnetic field
    :param showgeo: Show Geometry of the simulation before any excitation, defaults to True
    :param modevisulization: Visualize the Modes in the Photonic Crystal, default set to True
    :param findModes: Use Harminv to find the modes of the photonic crystal with the specified excitation signal
    :param input_freq: Input Frequency in Meep normalized units, do not have MaxQ_freq if you do want a specified frequency
    :param MaxQ_freq: Set input_freq variable to the highest Q frequency found from HarmInv, need findModes to be True
    :param resolution: Set resolution of the computational space, defaults to 64
    :param smoothing: Subpixel smoothing enabler, defaults to True
    :param ModeVolume: Calculate the Mode volume with local density of states, defaults to true
    """


    Magnetic_field = {mp.Hx, mp.Hy, mp.Hz}

    c = 3e8 # Speed of light for conversion later
    mp.verbosity(3)

    # Parameters for the computational cell: 
    pml_padding = length
    padding_to_pml = 4*length
    sx = 2*(len(ylengths))-1 + 2* nummirrors  # We will scale the dimensions of everything else by the length (for example we scaled the paddings above by the length)
    sy = 2*(pml_padding + padding_to_pml + width)/length  

    cell = mp.Vector3(sx,sy)

    # For the silicon we want to use and our approximate target wavelength is below: 
    eps_silicon = 12
    thickness = 220
    wavelength = 1540 

    # Defining our waveguide given the parameters, meep assumes periodic structure: 
    beam = mp.Block(size = (sx, width/length, thickness/length), material = mp.Medium(epsilon = eps_silicon) )
    holes = []
    for i in range(len(ylengths)): 
        holes.append(mp.Ellipsoid(center = mp.Vector3((-sx/2 + .5 + nummirrors) + i, 0, 0), size = mp.Vector3(xlength, ylengths[i], thickness)/length, e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1),
                          material = mp.Medium(epsilon = 1)))
        holes.append(mp.Ellipsoid(center = mp.Vector3((+sx/2 - .5 - nummirrors) - i, 0, 0), size = mp.Vector3(xlength, ylengths[i], thickness)/length, e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1),
                          material = mp.Medium(epsilon = 1)))

    for i in range(nummirrors): 
        holes.append(mp.Ellipsoid( center = mp.Vector3((-sx/2 + 0.5) + i , 0, 0 ), size = mp.Vector3(xlength, mirrorlength, thickness)/length, e1 = mp.Vector3(1,0,0), 
                     e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
        holes.append(mp.Ellipsoid( center = mp.Vector3((+sx/2 - 0.5) - i , 0, 0 ), size = mp.Vector3(xlength, mirrorlength, thickness)/length, e1 = mp.Vector3(1,0,0), 
                     e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))

    geometry = [beam]+holes

    # Other things for simulation: It is to note that say we want to run for a certian number of periods, then we would divide our time by the frequency below
    fcenter = length/wavelength           # In meep units, the frequency is specified by 1/lambda according to https://meep.readthedocs.io/en/latest/Introduction/#units-in-meep
    df = .3

    # Broad source to excite modes
    source = mp.Source(mp.GaussianSource(fcenter, fwidth = df), excitation, mp.Vector3(0, 0))
    pml_layers = [mp.PML(pml_padding/length, direction = mp.Y), mp.PML(pml_padding/length,direction = mp.X)]


    if excitation in Magnetic_field:
        sym = [mp.Mirror(mp.X, phase =-1), mp.Mirror(mp.Y, phase = -1)]
    else: 
        sym = [mp.Mirror(mp.X, phase = 1), mp.Mirror(mp.Y, phase = 1)]

    decay_period = 20
    decay_tolerance = 1e-6

    if Bulk: 
        simulation = mp.Simulation(
        cell_size = cell, 
        boundary_layers = pml_layers, 
        eps_averaging=smoothing, 
        geometry = [mp.Block(size = cell, material = mp.Medium(epsilon = eps_silicon))], 
        sources = [source], 
        resolution = resolution,
        symmetries = sym,
        )

        if showgeo: 
            f = plt.figure()
            simulation.plot2D(ax = f.gca())
            f.gca().xaxis.set_visible(False)
            f.suptitle('Geometry of Simulation')
            plt.axvline(x = -(sx/2) + nummirrors , color = 'k', linestyle = '--')
            plt.axvline(x = (sx/2) - nummirrors , color = 'k', linestyle = '--')
            plt.show()
            plt.close() 

        simulation.run(
        mp.dft_ldos(fcenter, 0, 1), 
        until_after_sources=mp.stop_when_fields_decayed(
            decay_period, excitation, mp.Vector3(), decay_tolerance
            )
        )

        return simulation.ldos_data[0]

    simulation = mp.Simulation(
        cell_size = cell, 
        boundary_layers = pml_layers, 
        eps_averaging=smoothing, 
        geometry = geometry, 
        sources = [source], 
        resolution = resolution,
        symmetries = sym
        )

    if showgeo: 
        f = plt.figure()
        simulation.plot2D(ax = f.gca())
        f.gca().xaxis.set_visible(False)
        f.suptitle('Geometry of Simulation')
        plt.axvline(x = -(sx/2) + nummirrors , color = 'k', linestyle = '--')
        plt.axvline(x = (sx/2) - nummirrors , color = 'k', linestyle = '--')
        plt.show()
        plt.close() 

    simulation.run(
        mp.dft_ldos(fcenter, 0, 1), 
        until_after_sources=mp.stop_when_fields_decayed(
            decay_period, excitation, mp.Vector3(), decay_tolerance
            )
        )
       

    return simulation.ldos_data[0]