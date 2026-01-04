

# The goal of this script is to create a function that takes in user inputted parameters for a cell (its width, length, and radius of holes) and outputs the band diagram for that strucutre. It is to note that this script will be made with the material being Silicon in a vacuum. 

# Importing Libraries: 

import meep as mp
import matplotlib.pyplot as plt
import numpy as np 
from IPython.display import Video 



# Maxwell's Equations are scale invariant, because of this I will choose the characteristic length scale to be a tenth of a micron



def unitcellbandstrucuture(width, length, diameter):
    # width - width of our beam in nanometers
    # length - length of our beam in nanometers
    # diameter - diameter of the holes in patterned into the beam in nanometers

    # Parameters for the computational cell: 
    pml_padding = length  
    padding_to_pml = 4*length
    sx = 1       #Respective sizes along given axis for our computational zone
    sy = 2*(pml_padding + padding_to_pml + width)/length  
    res = 50     

    cell = mp.Vector3(sx,sy)

    # For the silicon we want to use and our approximate target wavelength is below: 
    eps_silicon = 12
    thickness = 220
    wavelength = 1540 

    # Defining our waveguide given the parameters, meep assumes periodic structure: 
    beam = mp.Block(size = (1e20, width/length, 1e20), material = mp.Medium(epsilon = eps_silicon) )
    holes = mp.Cylinder(radius = (diameter/2)/length, material = mp.Medium(epsilon = 1))

    geometry = [beam,holes]

    # Other things for simulation: It is to note that say we want to run for a certian number of periods, then we would divide our time by the frequency below
    fcenter = length/wavelength           # In meep units, the frequency is specified by 1/lambda according to https://meep.readthedocs.io/en/latest/Introduction/#units-in-meep
    df = .5*fcenter

    sym = mp.Mirror(direction=mp.Y, phase=-1)
    # Broad source to excite modes
    source = mp.Source(mp.GaussianSource(fcenter, fwidth = df), mp.Hz, mp.Vector3(0, 0))
    pml_layers = [mp.PML(pml_padding/length, direction =mp.Y)]

    simulation = mp.Simulation(
        cell_size = cell, 
        boundary_layers = pml_layers, 
        geometry = geometry, 
        sources = [source], 
        resolution = res, 
        symmetries=[sym]
    )

    f = plt.figure(dpi=200)
    simulation.plot2D(ax = f.gca())
    f.gca().xaxis.set_visible(False)
    f.suptitle('Geometry of Simulation')
    plt.show()


    h = mp.Harminv(mp.Hz, mp.Vector3(), fcenter, df)


    simulation.run(
    mp.at_beginning(mp.output_epsilon), 
    mp.after_sources(h),
    until_after_sources = 400
)

    simulation.sources = [
   mp.Source(mp.GaussianSource(fcenter, df), mp.Hz, mp.Vector3())
]   

    simulation.restart_fields()
    k_interp = 19
    kpts = mp.interpolate(k_interp, [mp.Vector3(0), mp.Vector3(.5)])
    all_freqs = simulation.run_k_points(300, kpts)

    kx = [k.x for k in kpts]
    fig = plt.figure(dpi=100, figsize=(5, 5))
    ax = plt.subplot(111)
    for i in range(len(all_freqs)):
        for ii in range(len(all_freqs[i])):
            plt.scatter(kx[i], np.real(all_freqs[i][ii]), color="b")

    ax.fill_between(kx, kx, 1.0, interpolate=True, color="gray", alpha=0.3)
    plt.xlim(0, .5)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xlabel("$k_x(2\pi)$")
    plt.ylabel("$\omega(2\pi c)$")
    plt.tight_layout()
    plt.show()