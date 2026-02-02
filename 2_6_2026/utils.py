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
import os


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

    N = np.arange(1,n)
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

def QuadraticTaperwithMirrors(endLength: float, startLength: float, width: float, i: int):
    """
    Function for creating taper to ending length
    :param endLength: Ending length of taper, the yintercept of the quadratic
    :param startLength: Mirror Length
    :param width: The factor in front of our quadratic term, determines the width of the quadratic
    :param i: index of the hole, determines the length of that hole
    """

    length = endLength + (startLength/endLength) * width * i**2
    if length > startLength: 
        return startLength
    return length 


    
def defectandmirrors(width: float, length:float, mirrorxlength: float, mirrorlength: float,
                    ylength: float, xlength: float, nummirrors: int = 8, numdefects: int = 9, 
                    excitation = mp.Hz, resolution: int = 64, smoothing: bool = True,
                    showgeo: bool = True, modevisulization: bool = True, findModes: bool = True, 
                    ModeVolume:bool = True, dir: str ='', lorentz: bool = False, 
                    ):
    """
    Docstring for defectandmirrors
    
    Parameters of "Unit Cell"/Mirrors: 
    :param width: Width of Unit Cell, how long it is in y-direction [nm]
    :param length: Length of Unit Cell, used as our characteristic length (x-dir) [nm]
    :param mirrorxlength: Minor axis of mirrors, in x-direction [nm]
    :param mirrorlength: Major axis length of Mirror, in y-direction [nm] 

    Parameters of Defect Holes:
    :param ylength: Middle Defect's y-length [nm]
    :param xlength: Middle Defect's x-length [nm]

    Parameters for Crystal: 
    :param nummirrors: Number of mirrors on one side
    :param numdefects: Number of defects in total (Best to use odd number here)

    Parameters for Simulation:
    :param excitation: Meep Excitation Signal used in simulations, defaulted to Z-component magnetic field
    :param resolution: Set resolution of the computational space, defaults to 64
    :param smoothing: Subpixel smoothing enabler, defaults to True

    Extra Features: 
    :param showgeo: Show/Save Geometry of the simulation before any excitation, defaults to True
    :param modevisulization: Visualize a Mode in the Photonic Crystal, default set to True (Change accordingly to which mode you want)
    :param findModes: Use Harminv to find the modes of the photonic crystal with the specified excitation signal
    :param ModeVolume: Enables calculation of the Mode volume with modal_volume_in_box
    :param dir: String for directory to save all the images and text files
    :param lorentz: Calculates the Lorentzian tranmission curve for a mode
    """


    # Below are some constants useful for other features later
    Magnetic_field = {mp.Hx, mp.Hy, mp.Hz}
    c = 3e8 
    conv = (c/(length*1e3)) 
    purcell_prefix = (3/(4*np.pi**2)) 
    eps_silicon = 12
    refraction_silicon = 3.4
    thickness = 220
    wavelength = 1540 


    #Change Verbosity to how much information you want the simulations to output during runtime
    mp.verbosity(1)
    if mp.am_master(): 
        os.makedirs(dir, exist_ok=True) #Useful for parallel meep


    # Parameters for the computational cell: 
    pml_padding = length
    beam_padding = 2                    # One unit cell length each side
    air_padding = (wavelength/length)   # conversion of wavelength into units of characteristic length (name sucks)
    pml_padding_x = 4 * air_padding     # One side, four wavelengths long in characteristic lengths
    padding_to_pml = 4*length
    sx = numdefects + 2 * nummirrors + beam_padding 
    sy = 2*(pml_padding + padding_to_pml + width)/length  
    cell = mp.Vector3(sx + 2 * pml_padding_x,sy)


    #Defining taper for defect region holes:
    taper_length = numdefects // 2
    verticalTapering = tapering(mirrorlength, ylength, taper_length)
    verticalTapering_reverse = np.flip(verticalTapering)
    horizontalTapering = tapering(mirrorxlength, xlength, taper_length)    #We will manually define the middle defect's geo
    horizontalTapering_reverse = np.flip(horizontalTapering)

    
    # Defining our crystal:
    beam = mp.Block(size = (sx, width/length, thickness/length), material = mp.Medium(epsilon = eps_silicon) )
    holes = [mp.Ellipsoid(center = mp.Vector3(),
                        size = mp.Vector3(xlength, ylength, thickness)/length, 
                        e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1))] #Center defect
    for i in range(taper_length + nummirrors):
        if i+1 < taper_length: 
            holes.append(mp.Ellipsoid(center = mp.Vector3(i+1, 0, 0), 
                                      size = mp.Vector3(horizontalTapering_reverse[i], verticalTapering_reverse[i], thickness)/length,
                                      e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
            holes.append(mp.Ellipsoid(center = mp.Vector3(-i-1, 0, 0), 
                                      size = mp.Vector3(horizontalTapering_reverse[i], verticalTapering_reverse[i], thickness)/length,
                                      e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
        else: 
              holes.append(mp.Ellipsoid(center = mp.Vector3(i+1, 0, 0), 
                                      size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length,
                                      e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))        
              holes.append(mp.Ellipsoid(center = mp.Vector3(-i-1, 0, 0), 
                                      size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length,
                                      e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))                            
    
    #for i in range(len(ylengths)): 
        #holes.append(mp.Ellipsoid(center = mp.Vector3((-(sx)/2 + .5  + nummirrors) + i, 0, 0), size = mp.Vector3(mirrorxlength, ylengths[i], thickness)/length, e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1),
        #                  material = mp.Medium(epsilon = 1)))
        #holes.append(mp.Ellipsoid(center = mp.Vector3((+(sx)/2 - .5  - nummirrors) - i, 0, 0), size = mp.Vector3(mirrorxlength, ylengths[i], thickness)/length, e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1),
        #                  material = mp.Medium(epsilon = 1)))

    #for i in range(nummirrors): 
        #holes.append(mp.Ellipsoid( center = mp.Vector3((-(sx)/2 + .5 ) + i , 0, 0 ), size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length, e1 = mp.Vector3(1,0,0), 
        #             e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
        #holes.append(mp.Ellipsoid( center = mp.Vector3((+(sx)/2 - .5 ) - i , 0, 0 ), size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length, e1 = mp.Vector3(1,0,0), 
        #             e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))

    geometry = [beam]+holes
    
    
    # Other things for simulation:
    fcenter = length/wavelength  
    df = .07
    source = mp.Source(mp.GaussianSource(fcenter, fwidth = df), excitation, center = mp.Vector3(0, 0), size = mp.Vector3(0,0))
    pml_layers = [mp.PML(pml_padding_x, direction = mp.X),mp.PML(pml_padding/length, direction = mp.Y)]
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


    #mp master for parallel meep compatibility    
    if showgeo and mp.am_master(): 
        f = plt.figure()
        simulation.plot2D(ax = f.gca())
        f.gca().xaxis.set_visible(False)
        f.suptitle('Geometry of Simulation')
        plt.axvline(x = taper_length + .5 , color = 'k', linestyle = '--')
        plt.axvline(x = -taper_length - .5, color = 'k', linestyle = '--')
        plt.savefig(os.path.join(dir,"geometry.png"), dpi=300)
        plt.close()
        

    # Harminv to find modes:
    freqs = []
    Qs = []
    V_mode = 0
    if findModes:
        h = mp.Harminv(excitation, mp.Vector3(), fcenter, df)
        simulation.run(
            mp.after_sources(h),
            until_after_sources=2000
        )
        # Parallel Meep compatibility
        if mp.am_master():
            freqs = [m.freq for m in h.modes] 
            Qs = [ min(m.Q,10**7) for m in h.modes ]
            decay = [m.decay for m in h.modes]
            amp = [m.decay for m in h.modes]
            err = [m.err for m in h.modes]
            # Filter Out bad/unreliable data
            Useful_data = [(f,q,d,a,e) for f,q,d,a,e in zip(freqs, Qs, decay, amp, err) if abs(d) > abs(e)*10**3]
            if Useful_data: 
                freqs, Qs, decay, amp, err = zip(*Useful_data)
                freqs, Qs, decay, amp, err = list(freqs), list(Qs), list(decay), list(amp), list(err)
            else: 
                freqs, Qs, decay, amp, err = [], [], [], [], []
            

        # Calculate mode volume if requested
        if ModeVolume:
            try:      
                V_mode = simulation.modal_volume_in_box(
                    mp.Volume(center=mp.Vector3(), 
                    size=cell))
            except Exception as e:
                if mp.am_master():
                    print(f"Mode volume calculation failed: {e}")
                    V_mode = 0
        
        
        # Find maximum Q and corresponding frequency
        if len(Qs) > 0:
            maxq = max(Qs)
            ind = Qs.index(maxq)
            freq_max = freqs[ind]
            freq_maxq = freq_max*conv
            wavelength_max_nm = (c / (freq_maxq * 1e12)) * 1e9
            #Print results (Uncomment if you want in terminal)
            #print(f'\n{"="*60}')
            #print(f'HARMINV RESULTS:')
            #print(f'{"="*60}')
            #print(f'Number of modes found: {len(freqs)}')
            #print(f'Highest Q factor: {maxq:.2f}')
            #print(f'Frequency of highest Q: {freq_maxq:.4f} THz')
            #if ModeVolume:
            #    print(f'Mode volume: {V_mode:.6f}')
            #    print(f'{"="*60}\n')
            #Save mode data to file
            with open(os.path.join(dir,"mode_analysis.txt"), "w") as f:
                f.write("MODE ANALYSIS\n")
                f.write("="*100 + "\n\n")
                f.write("Simulation Parameters:\n")
                f.write(f"  Width: {width} nm\n")
                f.write(f"  Length: {length} nm\n")
                f.write(f"  Number of defect holes: {numdefects}\n")
                f.write(f"  Number of mirrors: {nummirrors}\n")
                f.write(f"  Resolution: {resolution}\n\n")
                f.write("="*150 + "\n")
                f.write("DETECTED MODES:\n")
                f.write("="*150 + "\n")
                f.write(f"{'Mode':<8} {'Frequency (THz)':<20} {'Freq (Meep)':<20} {'Decay Rate':<40} {'Error':<20} {'Q Factor':<15} {'Wavelength (nm)':<15}\n")
                f.write("-"*150 + "\n")
                for i, (fr,q,d,e) in enumerate(zip(freqs, Qs, decay, err)):
                    freq_thz = fr * conv
                    wavelength_nm = (c / (freq_thz * 1e12)) * 1e9
                    f.write(f"{i+1:<8} {freq_thz:<20.6f} {fr:<20.6f} {d:<20.10f} {e:<40.10f} {q:<15.2f} {wavelength_nm:<15.2f}\n")
                f.write("\n" + "="*150 + "\n")
                f.write(f"HIGHEST Q MODE:\n")
                f.write("="*150 + "\n")
                f.write(f"  Mode index: {ind+1}\n")
                f.write(f"  Frequency: {freq_maxq:.6f} THz\n")
                f.write(f'  Freq (Meep): {freq_max:.6f}\n')
                f.write(f"  Q factor: {maxq:.2f}\n")
                f.write(f"  Wavelength: {(c/(freq_maxq*1e12))*1e9:.2f} nm\n")
                if ModeVolume:
                    f.write(f"  Mode volume: {(V_mode*length**2)/(wavelength_max_nm/refraction_silicon)**2:.6f} (lambda/n)^2\n")
                    purcell_factor = purcell_prefix * ( wavelength_max_nm/refraction_silicon)**3 * (maxq/ (V_mode * length**2 * thickness) )
                    f.write(f"  Purcell Factor: {purcell_factor}\n")
                # Save raw data in CSV format for easy importing (Change Later)
                import csv
                with open(os.path.join(dir,"mode_data.csv"), "w", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Mode", "Frequency_THz", 'Freq_Meep' "Q_Factor", "Wavelength_nm"])
                    for i, (freq, q) in enumerate(zip(freqs, Qs)):
                        freq_thz = freq * conv
                        wavelength_nm = (c / (freq_thz * 1e12)) * 1e9
                        writer.writerow([i+1, freq_thz, freq, q, wavelength_nm])
        else:
            print("WARNING: No Good Modes")


    if modevisulization: 
        #Narrow source to excite the mode you want (change frequency according to what mode you want to excite)
        source_new = mp.Source(mp.GaussianSource(freq_max, fwidth = .05), excitation, mp.Vector3(0, 0), size = mp.Vector3(0,))
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
            # Create a list of field components and their names
            field_components = [
                (mp.Ex, "Ex"),
                (mp.Ey, "Ey"),
                (mp.Ez, "Ez"),
                (mp.Hx, "Hx"),
                (mp.Hy, "Hy"),
                (mp.Hz, "Hz")
            ]            
            for field, field_name in field_components:
                f = plt.figure(dpi=150)
                plt.axvline(x = taper_length + .5, color = 'k', linestyle = '--')
                plt.axvline(x = -taper_length -.5 , color = 'k', linestyle = '--')
                sim2.plot2D(ax=f.gca(), fields=field)
                plt.title(f'{field_name} Component')
                plt.xlabel('x (normalized units)')
                plt.ylabel('y (normalized units)')
                plt.savefig(os.path.join(dir, f"{field_name}.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
    
    if lorentz: 
        # Used for optimization purposes (change freqs[i] accordingly)
        detuning = (freqs[0] - fcenter)/fcenter
        kappa_factors = 2/(maxq)
        transmission = kappa_factors ** 2 / (detuning ** 2 + kappa_factors ** 2)
        return detuning**2
            

    return Qs, freqs, V_mode, purcell_factor


def defectandmirrors2(width: float, length:float, mirrorxlength: float, mirrorlength: float,
                    ylength: float, xlength: float, widthx: float, widthy: float, numholes: int = 17, 
                    excitation = mp.Hz, resolution: int = 64, smoothing: bool = True,
                    showgeo: bool = True, modevisulization: bool = True, findModes: bool = True, 
                    ModeVolume:bool = True, dir: str ='', lorentz: bool = False 
                    ):
    """
    Docstring for defectandmirrors
    
    Parameters of "Unit Cell"/Mirrors: 
    :param width: Width of Unit Cell, how long it is in y-direction [nm]
    :param length: Length of Unit Cell, used as our characteristic length (x-dir) [nm]
    :param mirrorxlength: Minor axis of mirrors, in x-direction [nm]
    :param mirrorlength: Major axis length of Mirror, in y-direction [nm] 

    Parameters of Defect Holes:
    :param ylength: Middle Defect's y-length [nm]
    :param xlength: Middle Defect's x-length [nm]

    Parameters for Crystal: 
    :param numholes: number of holes for one side of the taper

    Parameters for Simulation:
    :param excitation: Meep Excitation Signal used in simulations, defaulted to Z-component magnetic field
    :param resolution: Set resolution of the computational space, defaults to 64
    :param smoothing: Subpixel smoothing enabler, defaults to True

    Extra Features: 
    :param showgeo: Show/Save Geometry of the simulation before any excitation, defaults to True
    :param modevisulization: Visualize a Mode in the Photonic Crystal, default set to True (Change accordingly to which mode you want)
    :param findModes: Use Harminv to find the modes of the photonic crystal with the specified excitation signal
    :param ModeVolume: Enables calculation of the Mode volume with modal_volume_in_box
    :param dir: String for directory to save all the images and text files
    :param lorentz: Calculates the Lorentzian tranmission curve for a mode
    """


    # Below are some constants useful for other features later
    Magnetic_field = {mp.Hx, mp.Hy, mp.Hz}
    c = 3e8 
    conv = (c/(length*1e3)) 
    purcell_prefix = (3/(4*np.pi**2)) 
    eps_silicon = 12
    refraction_silicon = 3.4
    thickness = 220
    wavelength = 1540 


    if excitation in Magnetic_field:
        # Ez = 0 -> TE
        sym = [mp.Mirror(mp.X, phase =-1), mp.Mirror(mp.Y, phase = -1)]
        refraction_silicon = 2.736674116084838
        beta = 11109379.467855845
        eps_silicon = refraction_silicon**2
    else: 
        # Hz = 0 -> TM
        sym = [mp.Mirror(mp.X, phase = 1), mp.Mirror(mp.Y, phase = 1)]
        refraction_silicon = 1.5464070772068217 # Found retrospectively for now
        beta = 6277555.274665322
        eps_silicon = refraction_silicon**2

    #Change Verbosity to how much information you want the simulations to output during runtime
    mp.verbosity(1)
    if mp.am_master(): 
        os.makedirs(dir, exist_ok=True) #Useful for parallel meep


    # Parameters for the computational cell: 
    pml_padding = length
    beam_padding = 2                    # One unit cell length each side
    air_padding = (wavelength/length)   # conversion of wavelength into units of characteristic length (name sucks)
    pml_padding_x = 4 * air_padding     # One side, four wavelengths long in characteristic lengths
    padding_to_pml = 4*length
    sx = 1 + 2 * numholes + beam_padding 
    sy = 2*(pml_padding + padding_to_pml + width)/length  
    cell = mp.Vector3(sx + 2 * pml_padding_x,sy)


    #Defining taper for defect region holes:
    #taper_length = numdefects // 2
    #verticalTapering = tapering(mirrorlength, ylength, taper_length)
    #verticalTapering_reverse = np.flip(verticalTapering)
    #horizontalTapering = tapering(mirrorxlength, xlength, taper_length)    #We will manually define the middle defect's geo
    #horizontalTapering_reverse = np.flip(horizontalTapering)

    
    # Defining our crystal:
    beam = mp.Block(size = (sx, width/length, thickness/length), material = mp.Medium(epsilon = eps_silicon) )
    holes = [mp.Ellipsoid(center = mp.Vector3(),
                        size = mp.Vector3(xlength, ylength, thickness)/length, 
                        e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1))] #Center defect
    
    mirror_indicies_vert = []
    mirror_indicies_horiz = []
    verts = [ylength]
    horiz =[xlength]
    for i in range(numholes): 
        vertlength = QuadraticTaperwithMirrors(endLength=ylength, startLength=mirrorlength, width=widthy, i=i+1)
        horzlength = QuadraticTaperwithMirrors(endLength=xlength, startLength=mirrorxlength, width=widthx, i=i+1)
        verts.append(vertlength)
        horiz.append(horzlength)
        if vertlength == mirrorlength: #or horzlength == mirrorxlength: 
            mirror_indicies_vert.append(i)
        if horzlength == mirrorxlength:
            mirror_indicies_horiz.append(i)
        holes.append(mp.Ellipsoid(center = mp.Vector3(i+1,0,0), 
                                  size = mp.Vector3(horzlength, vertlength, thickness)/length, 
                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
        holes.append(mp.Ellipsoid(center = mp.Vector3(-i-1,0,0), 
                                  size = mp.Vector3(horzlength, vertlength, thickness)/length, 
                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
    taper_length_vert = mirror_indicies_vert[0] if mirror_indicies_vert else 1
    taper_length_horiz = mirror_indicies_horiz[0] if mirror_indicies_horiz else 1
    taper_length = min(taper_length_horiz, taper_length_vert)
    if mp.am_master(): 
        f = plt.figure()
        plt.plot(range(numholes+1), verts)
        f.gca().xaxis.set_visible(False)
        f.suptitle('Vertical Quadratic')
        plt.savefig(os.path.join(dir,"VerticalQuadratic.png"), dpi=300)
        plt.close()
        f = plt.figure()
        plt.plot(range(numholes+ 1) , horiz)
        f.gca().xaxis.set_visible(False)
        f.suptitle('Horizontal Quadratic')
        plt.savefig(os.path.join(dir,"HorizontalQuadratic.png"), dpi=300)
        plt.close()
    #for i in range(taper_length + nummirrors):
    #    if i+1 < taper_length: 
    #        holes.append(mp.Ellipsoid(center = mp.Vector3(i+1, 0, 0), 
    #                                  size = mp.Vector3(horizontalTapering_reverse[i], verticalTapering_reverse[i], thickness)/length,
    #                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
    #        holes.append(mp.Ellipsoid(center = mp.Vector3(-i-1, 0, 0), 
    #                                 size = mp.Vector3(horizontalTapering_reverse[i], verticalTapering_reverse[i], thickness)/length,
    #                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
    #    else: 
    #          holes.append(mp.Ellipsoid(center = mp.Vector3(i+1, 0, 0), 
    #                                  size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length,
    #                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))        
    #          holes.append(mp.Ellipsoid(center = mp.Vector3(-i-1, 0, 0), 
    #                                  size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length,
    #                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))                            
    
    #for i in range(len(ylengths)): 
        #holes.append(mp.Ellipsoid(center = mp.Vector3((-(sx)/2 + .5  + nummirrors) + i, 0, 0), size = mp.Vector3(mirrorxlength, ylengths[i], thickness)/length, e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1),
        #                  material = mp.Medium(epsilon = 1)))
        #holes.append(mp.Ellipsoid(center = mp.Vector3((+(sx)/2 - .5  - nummirrors) - i, 0, 0), size = mp.Vector3(mirrorxlength, ylengths[i], thickness)/length, e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1),
        #                  material = mp.Medium(epsilon = 1)))

    #for i in range(nummirrors): 
        #holes.append(mp.Ellipsoid( center = mp.Vector3((-(sx)/2 + .5 ) + i , 0, 0 ), size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length, e1 = mp.Vector3(1,0,0), 
        #             e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
        #holes.append(mp.Ellipsoid( center = mp.Vector3((+(sx)/2 - .5 ) - i , 0, 0 ), size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length, e1 = mp.Vector3(1,0,0), 
        #             e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))

    geometry = [beam]+holes
    
    
    # Other things for simulation:
    fcenter = length/wavelength  
    df = .2
    source = mp.Source(mp.GaussianSource(fcenter, fwidth = df), excitation, center = mp.Vector3(0, 0), size = mp.Vector3(0,0))
    pml_layers = [mp.PML(pml_padding_x, direction = mp.X),mp.PML(pml_padding/length, direction = mp.Y)]
    


    simulation = mp.Simulation(
        cell_size = cell, 
        boundary_layers = pml_layers, 
        eps_averaging=smoothing, 
        geometry = geometry, 
        sources = [source], 
        resolution = resolution,
        symmetries = sym
        )


    #mp master for parallel meep compatibility    
    if showgeo and mp.am_master(): 
        f = plt.figure()
        simulation.plot2D(ax = f.gca())
        f.gca().xaxis.set_visible(False)
        f.suptitle('Geometry of Simulation')
        plt.axvline(x = taper_length_vert + .5 , color = 'b', linestyle = '--')
        plt.axvline(x = -taper_length_vert - .5, color = 'b', linestyle = '--')
        plt.axvline(x = taper_length_horiz + .5 , color = 'k', linestyle = '--')
        plt.axvline(x = -taper_length_horiz - .5, color = 'k', linestyle = '--')
        plt.savefig(os.path.join(dir,"geometry.png"), dpi=300)
        plt.close()
        

    # Harminv to find modes:
    freqs = []
    Qs = []
    V_mode = 0
    if findModes:
        h = mp.Harminv(excitation, mp.Vector3(), fcenter, df)
        simulation.run(
            mp.after_sources(h),
            until_after_sources=6000
        )
        # Parallel Meep compatibility
        if mp.am_master():
            freqs = [m.freq for m in h.modes] 
            Qs = [ min(m.Q,10**7) for m in h.modes ]
            decay = [m.decay for m in h.modes]
            amp = [m.decay for m in h.modes]
            err = [m.err for m in h.modes]
            # Filter Out bad/unreliable data
            Useful_data = [(f,q,d,a,e) for f,q,d,a,e in zip(freqs, Qs, decay, amp, err) if abs(d) > abs(e)*10**3 or abs(e)<10**-7]
            if Useful_data: 
                freqs, Qs, decay, amp, err = zip(*Useful_data)
                freqs, Qs, decay, amp, err = list(freqs), list(Qs), list(decay), list(amp), list(err)
            else: 
                freqs, Qs, decay, amp, err = [], [], [], [], []
            

        # Calculate mode volume if requested
        if ModeVolume:
            try:      
                V_mode = simulation.modal_volume_in_box(
                    mp.Volume(center=mp.Vector3(), 
                    size=cell))
            except Exception as e:
                if mp.am_master():
                    print(f"Mode volume calculation failed: {e}")
                    V_mode = 0
        
        
        # Find maximum Q and corresponding frequency
        if len(Qs) > 0:
            maxq = max(Qs)
            ind = Qs.index(maxq)
            freq_max = freqs[ind]
            freq_maxq = freq_max*conv
            wavelength_max_nm = (c / (freq_maxq * 1e12)) * 1e9
            #Print results (Uncomment if you want in terminal)
            #print(f'\n{"="*60}')
            #print(f'HARMINV RESULTS:')
            #print(f'{"="*60}')
            #print(f'Number of modes found: {len(freqs)}')
            #print(f'Highest Q factor: {maxq:.2f}')
            #print(f'Frequency of highest Q: {freq_maxq:.4f} THz')
            #if ModeVolume:
            #    print(f'Mode volume: {V_mode:.6f}')
            #    print(f'{"="*60}\n')
            #Save mode data to file
            with open(os.path.join(dir,"mode_analysis.txt"), "w") as f:
                f.write("MODE ANALYSIS\n")
                f.write("="*100 + "\n\n")
                f.write("Simulation Parameters:\n")
                f.write(f"  Width: {width} nm\n")
                f.write(f"  Length: {length} nm\n")
                f.write(f"  Number of defect holes: {2*(taper_length)+1}\n")
                f.write(f"  Number of mirrors: {2*(numholes - taper_length)}\n")
                f.write(f"  Resolution: {resolution}\n\n")
                f.write("="*150 + "\n")
                f.write("DETECTED MODES:\n")
                f.write("="*150 + "\n")
                f.write(f"{'Mode':<8} {'Frequency (THz)':<20} {'Freq (Meep)':<20} {'Decay Rate':<40} {'Error':<20} {'Q Factor':<15} {'Wavelength (nm)':<15}\n")
                f.write("-"*150 + "\n")
                for i, (fr,q,d,e) in enumerate(zip(freqs, Qs, decay, err)):
                    freq_thz = fr * conv
                    wavelength_nm = (c / (freq_thz * 1e12)) * 1e9
                    f.write(f"{i+1:<8} {freq_thz:<20.6f} {fr:<20.6f} {d:<20.10f} {e:<40.10f} {q:<15.2f} {wavelength_nm:<15.2f}\n")
                f.write("\n" + "="*150 + "\n")
                f.write(f"HIGHEST Q MODE:\n")
                f.write("="*150 + "\n")
                f.write(f"  Mode index: {ind+1}\n")
                f.write(f"  Frequency: {freq_maxq:.6f} THz\n")
                f.write(f'  Freq (Meep): {freq_max:.6f}\n')
                f.write(f"  Q factor: {maxq:.2f}\n")
                f.write(f"  Wavelength: {(c/(freq_maxq*1e12))*1e9:.2f} nm\n")
                if ModeVolume:
                    k0 = freq_maxq*2*np.pi / c 
                    h = np.sqrt(refraction_silicon**2 * k0**2 - beta**2)
                    prefactor = 2*np.sin(h * thickness / 2) / (h*wavelength_max_nm*1e-9 /  (2*np.pi*refraction_silicon))
                    f.write(f"  Mode volume: {prefactor * (V_mode*length**2)/(wavelength_max_nm/refraction_silicon)**2:.6f} (lambda/n)^3\n")
                    purcell_factor = purcell_prefix * ( wavelength_max_nm/refraction_silicon)**3 * (maxq/ (V_mode * length**2 * thickness) )
                    f.write(f"  Extrapolated Purcell Factor: {purcell_factor}\n")
                f.write("\n" + "="*150 + "\n")
                f.write(f'QUADRATIC ARRAYS \n')
                f.write(f'Vertical lengths:{verts}\n')
                f.write(f'Horizontal Lengths: {horiz}\n')
                # Save raw data in CSV format for easy importing (Change Later)
                import csv
                with open(os.path.join(dir,"mode_data.csv"), "w", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Mode", "Frequency_THz", 'Freq_Meep' "Q_Factor", "Wavelength_nm"])
                    for i, (freq, q) in enumerate(zip(freqs, Qs)):
                        freq_thz = freq * conv
                        wavelength_nm = (c / (freq_thz * 1e12)) * 1e9
                        writer.writerow([i+1, freq_thz, freq, q, wavelength_nm])
        else:
            print("WARNING: No Good Modes")


    if modevisulization: 
        #Narrow source to excite the mode you want (change frequency according to what mode you want to excite)
        source_new = mp.Source(mp.GaussianSource(freq_max, fwidth = .05), excitation, mp.Vector3(0, 0), size = mp.Vector3(0,))
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
            # Create a list of field components and their names
            field_components = [
                (mp.Ex, "Ex"),
                (mp.Ey, "Ey"),
                (mp.Ez, "Ez"),
                (mp.Hx, "Hx"),
                (mp.Hy, "Hy"),
                (mp.Hz, "Hz")
            ]            
            for field, field_name in field_components:
                f = plt.figure(dpi=150)
                plt.axvline(x = taper_length + .5, color = 'k', linestyle = '--')
                plt.axvline(x = -taper_length -.5 , color = 'k', linestyle = '--')
                sim2.plot2D(ax=f.gca(), fields=field)
                plt.title(f'{field_name} Component')
                plt.xlabel('x (normalized units)')
                plt.ylabel('y (normalized units)')
                plt.savefig(os.path.join(dir, f"{field_name}.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
    
    if lorentz: 
        # Used for optimization purposes (change freqs[i] accordingly)
        detuning = (freqs[0] - fcenter)/fcenter
        kappa_factors = 2/(maxq)
        transmission = kappa_factors ** 2 / (detuning ** 2 + kappa_factors ** 2)
        return detuning**2
            

    return Qs, freqs, V_mode, purcell_factor

def DandMirrors3D(width: float, length:float, mirrorxlength: float, mirrorlength: float,
                    ylength: float, xlength: float, widthx: float, widthy: float, numholes: int = 17, 
                    excitation = mp.Hz, resolution: int = 64, smoothing: bool = True,
                    showgeo: bool = True, modevisulization: bool = True, findModes: bool = True, 
                    ModeVolume:bool = True, dir: str ='', lorentz: bool = False 
                    ):
    """
    Docstring for defectandmirrors
    
    Parameters of "Unit Cell"/Mirrors: 
    :param width: Width of Unit Cell, how long it is in y-direction [nm]
    :param length: Length of Unit Cell, used as our characteristic length (x-dir) [nm]
    :param mirrorxlength: Minor axis of mirrors, in x-direction [nm]
    :param mirrorlength: Major axis length of Mirror, in y-direction [nm] 

    Parameters of Defect Holes:
    :param ylength: Middle Defect's y-length [nm]
    :param xlength: Middle Defect's x-length [nm]

    Parameters for Crystal: 
    :param numholes: number of holes for one side of the taper

    Parameters for Simulation:
    :param excitation: Meep Excitation Signal used in simulations, defaulted to Z-component magnetic field
    :param resolution: Set resolution of the computational space, defaults to 64
    :param smoothing: Subpixel smoothing enabler, defaults to True

    Extra Features: 
    :param showgeo: Show/Save Geometry of the simulation before any excitation, defaults to True
    :param modevisulization: Visualize a Mode in the Photonic Crystal, default set to True (Change accordingly to which mode you want)
    :param findModes: Use Harminv to find the modes of the photonic crystal with the specified excitation signal
    :param ModeVolume: Enables calculation of the Mode volume with modal_volume_in_box
    :param dir: String for directory to save all the images and text files
    :param lorentz: Calculates the Lorentzian tranmission curve for a mode
    """


    # Below are some constants useful for other features later
    Magnetic_field = {mp.Hx, mp.Hy, mp.Hz}
    c = 3e8 
    conv = (c/(length*1e3)) 
    purcell_prefix = (3/(4*np.pi**2)) 
    eps_silicon = 12
    refraction_silicon = 3.4
    thickness = 220
    wavelength = 1540 


    #Change Verbosity to how much information you want the simulations to output during runtime
    mp.verbosity(1)
    if mp.am_master(): 
        os.makedirs(dir, exist_ok=True) #Useful for parallel meep


    # Parameters for the computational cell: 
    pml_padding = length
    beam_padding = 2                    # One unit cell length each side
    air_padding = (wavelength/length)   # conversion of wavelength into units of characteristic length (name sucks)
    pml_padding_x = 4 * air_padding     # One side, four wavelengths long in characteristic lengths
    pml_padding_y = pml_padding_x/4
    pml_padding_z = pml_padding_x/4
    padding_to_pml = 3*air_padding
    sx = 1 + 2 * numholes + beam_padding 
    sy = 2*(padding_to_pml/12 + width)/length  
    sz = 2*(padding_to_pml/6 + thickness)/length  
    cell = mp.Vector3(sx + 2 * pml_padding_x,sy + 2*pml_padding_y, sz + 2*pml_padding_z)


    #Defining taper for defect region holes:
    #taper_length = numdefects // 2
    #verticalTapering = tapering(mirrorlength, ylength, taper_length)
    #verticalTapering_reverse = np.flip(verticalTapering)
    #horizontalTapering = tapering(mirrorxlength, xlength, taper_length)    #We will manually define the middle defect's geo
    #horizontalTapering_reverse = np.flip(horizontalTapering)

    
    # Defining our crystal:
    beam = mp.Block(size = (sx, width/length, thickness/length), material = mp.Medium(epsilon = eps_silicon) )
    holes = [mp.Ellipsoid(center = mp.Vector3(),
                        size = mp.Vector3(xlength, ylength, thickness)/length, 
                        e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1))] #Center defect
    
    mirror_indicies_vert = []
    mirror_indicies_horiz = []
    verts = [ylength]
    horiz =[xlength]
    for i in range(numholes): 
        vertlength = QuadraticTaperwithMirrors(endLength=ylength, startLength=mirrorlength, width=widthy, i=i+1)
        horzlength = QuadraticTaperwithMirrors(endLength=xlength, startLength=mirrorxlength, width=widthx, i=i+1)
        verts.append(vertlength)
        horiz.append(horzlength)
        if vertlength == mirrorlength: #or horzlength == mirrorxlength: 
            mirror_indicies_vert.append(i)
        if horzlength == mirrorxlength:
            mirror_indicies_horiz.append(i)
        holes.append(mp.Ellipsoid(center = mp.Vector3(i+1,0,0), 
                                  size = mp.Vector3(horzlength, vertlength, thickness)/length, 
                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
        holes.append(mp.Ellipsoid(center = mp.Vector3(-i-1,0,0), 
                                  size = mp.Vector3(horzlength, vertlength, thickness)/length, 
                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
    taper_length_vert = mirror_indicies_vert[0] if mirror_indicies_vert else 1
    taper_length_horiz = mirror_indicies_horiz[0] if mirror_indicies_horiz else 1
    taper_length = min(taper_length_horiz, taper_length_vert)
    if mp.am_master(): 
        f = plt.figure()
        plt.plot(range(numholes+1), verts)
        f.gca().xaxis.set_visible(False)
        f.suptitle('Vertical Quadratic')
        plt.savefig(os.path.join(dir,"VerticalQuadratic.png"), dpi=300)
        plt.close()
        f = plt.figure()
        plt.plot(range(numholes+ 1) , horiz)
        f.gca().xaxis.set_visible(False)
        f.suptitle('Horizontal Quadratic')
        plt.savefig(os.path.join(dir,"HorizontalQuadratic.png"), dpi=300)
        plt.close()
    #for i in range(taper_length + nummirrors):
    #    if i+1 < taper_length: 
    #        holes.append(mp.Ellipsoid(center = mp.Vector3(i+1, 0, 0), 
    #                                  size = mp.Vector3(horizontalTapering_reverse[i], verticalTapering_reverse[i], thickness)/length,
    #                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
    #        holes.append(mp.Ellipsoid(center = mp.Vector3(-i-1, 0, 0), 
    #                                 size = mp.Vector3(horizontalTapering_reverse[i], verticalTapering_reverse[i], thickness)/length,
    #                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
    #    else: 
    #          holes.append(mp.Ellipsoid(center = mp.Vector3(i+1, 0, 0), 
    #                                  size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length,
    #                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))        
    #          holes.append(mp.Ellipsoid(center = mp.Vector3(-i-1, 0, 0), 
    #                                  size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length,
    #                                  e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))                            
    
    #for i in range(len(ylengths)): 
        #holes.append(mp.Ellipsoid(center = mp.Vector3((-(sx)/2 + .5  + nummirrors) + i, 0, 0), size = mp.Vector3(mirrorxlength, ylengths[i], thickness)/length, e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1),
        #                  material = mp.Medium(epsilon = 1)))
        #holes.append(mp.Ellipsoid(center = mp.Vector3((+(sx)/2 - .5  - nummirrors) - i, 0, 0), size = mp.Vector3(mirrorxlength, ylengths[i], thickness)/length, e1 = mp.Vector3(1,0,0), e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1),
        #                  material = mp.Medium(epsilon = 1)))

    #for i in range(nummirrors): 
        #holes.append(mp.Ellipsoid( center = mp.Vector3((-(sx)/2 + .5 ) + i , 0, 0 ), size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length, e1 = mp.Vector3(1,0,0), 
        #             e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))
        #holes.append(mp.Ellipsoid( center = mp.Vector3((+(sx)/2 - .5 ) - i , 0, 0 ), size = mp.Vector3(mirrorxlength, mirrorlength, thickness)/length, e1 = mp.Vector3(1,0,0), 
        #             e2 = mp.Vector3(0,1,0), e3 = mp.Vector3(0,0,1)))

    geometry = [beam]+holes
    
    
    # Other things for simulation:
    fcenter = length/wavelength  
    df = .15
    source = mp.Source(mp.GaussianSource(fcenter, fwidth = df), excitation, center = mp.Vector3(), size = mp.Vector3())
    pml_layers = [mp.PML(pml_padding_x, direction = mp.X),mp.PML(pml_padding_y*(3/4), direction = mp.Y), mp.PML(pml_padding_z*(3/4), direction = mp.Z) ]
    if excitation in Magnetic_field:
        sym = [mp.Mirror(mp.X, phase =-1), mp.Mirror(mp.Y, phase = -1), mp.Mirror(mp.Z, phase = 1)]
    else: 
        sym = [mp.Mirror(mp.X, phase = 1), mp.Mirror(mp.Y, phase = 1), mp.Mirror(mp.Z, phase = -1)]


    simulation = mp.Simulation(
        cell_size = cell, 
        boundary_layers = pml_layers, 
        eps_averaging=smoothing, 
        geometry = geometry, 
        sources = [source], 
        resolution = resolution,
        symmetries = sym
        )


    #mp master for parallel meep compatibility    
    if showgeo and mp.am_master(): 
        f = plt.figure()
        simulation.plot2D(ax = f.gca(), output_plane = mp.Volume(center = mp.Vector3(0,0,0), size = mp.Vector3(sx + 2 * pml_padding_x,sy + 2 * pml_padding_y,0)))
        f.gca().xaxis.set_visible(True)
        f.suptitle('Geometry of Simulation (XY plane)')
        plt.axvline(x = taper_length_vert + .5 , color = 'b', linestyle = '--')
        plt.axvline(x = -taper_length_vert - .5, color = 'b', linestyle = '--')
        plt.axvline(x = taper_length_horiz + .5 , color = 'k', linestyle = '--')
        plt.axvline(x = -taper_length_horiz - .5, color = 'k', linestyle = '--')
        plt.savefig(os.path.join(dir,"geometryxy.png"), dpi=300)
        plt.close()

        f = plt.figure()
        simulation.plot2D(ax = f.gca(), output_plane = mp.Volume(center = mp.Vector3(0,0,0), size = mp.Vector3(0,sy + 2 * pml_padding_y,sz + 2 * pml_padding_z)))
        f.gca().xaxis.set_visible(True)
        f.suptitle('Geometry of Simulation (YZ plane)')
        plt.savefig(os.path.join(dir,"geometryyz.png"), dpi=300)
        plt.close()

        f = plt.figure()
        simulation.plot2D(ax = f.gca(), output_plane = mp.Volume(center = mp.Vector3(0,0,0), size = mp.Vector3(sx + 2 * pml_padding_x,0,sz + 2 * pml_padding_z)))
        f.gca().xaxis.set_visible(True)
        f.suptitle('Geometry of Simulation (XZ plane)')
        plt.savefig(os.path.join(dir,"geometryxz.png"), dpi=300)
        plt.close()

        

        

    # Harminv to find modes:
    freqs = []
    Qs = []
    V_mode = 0
    if findModes:
        h = mp.Harminv(excitation, mp.Vector3(), fcenter, df)
        simulation.run(
            mp.after_sources(h),
            until_after_sources=2500
        )
        # Parallel Meep compatibility
        if mp.am_master():
            freqs = [m.freq for m in h.modes] 
            Qs = [ min(m.Q,10**7) for m in h.modes ]
            decay = [m.decay for m in h.modes]
            amp = [m.decay for m in h.modes]
            err = [m.err for m in h.modes]
            # Filter Out bad/unreliable data
            Useful_data = [(f,q,d,a,e) for f,q,d,a,e in zip(freqs, Qs, decay, amp, err) if abs(d) > abs(e)*10**3 or abs(e)<10**-7]
            if Useful_data: 
                freqs, Qs, decay, amp, err = zip(*Useful_data)
                freqs, Qs, decay, amp, err = list(freqs), list(Qs), list(decay), list(amp), list(err)
            else: 
                freqs, Qs, decay, amp, err = [], [], [], [], []
            

        # Calculate mode volume if requested
        if ModeVolume:
            try:      
                V_mode = simulation.modal_volume_in_box(
                    mp.Volume(center=mp.Vector3(), 
                    size=cell))
            except Exception as e:
                if mp.am_master():
                    print(f"Mode volume calculation failed: {e}")
                    V_mode = 0
        
        
        # Find maximum Q and corresponding frequency
        if len(Qs) > 0:
            maxq = max(Qs)
            ind = Qs.index(maxq)
            freq_max = freqs[ind]
            freq_maxq = freq_max*conv
            wavelength_max_nm = (c / (freq_maxq * 1e12)) * 1e9
            #Print results (Uncomment if you want in terminal)
            #print(f'\n{"="*60}')
            #print(f'HARMINV RESULTS:')
            #print(f'{"="*60}')
            #print(f'Number of modes found: {len(freqs)}')
            #print(f'Highest Q factor: {maxq:.2f}')
            #print(f'Frequency of highest Q: {freq_maxq:.4f} THz')
            #if ModeVolume:
            #    print(f'Mode volume: {V_mode:.6f}')
            #    print(f'{"="*60}\n')
            #Save mode data to file
            with open(os.path.join(dir,"mode_analysis.txt"), "w") as f:
                f.write("MODE ANALYSIS\n")
                f.write("="*100 + "\n\n")
                f.write("Simulation Parameters:\n")
                f.write(f"  Width: {width} nm\n")
                f.write(f"  Length: {length} nm\n")
                f.write(f"  Number of defect holes: {2*(taper_length)+1}\n")
                f.write(f"  Number of mirrors: {2*(numholes - taper_length)}\n")
                f.write(f"  Resolution: {resolution}\n\n")
                f.write("="*150 + "\n")
                f.write("DETECTED MODES:\n")
                f.write("="*150 + "\n")
                f.write(f"{'Mode':<8} {'Frequency (THz)':<20} {'Freq (Meep)':<20} {'Decay Rate':<40} {'Error':<20} {'Q Factor':<15} {'Wavelength (nm)':<15}\n")
                f.write("-"*150 + "\n")
                for i, (fr,q,d,e) in enumerate(zip(freqs, Qs, decay, err)):
                    freq_thz = fr * conv
                    wavelength_nm = (c / (freq_thz * 1e12)) * 1e9
                    f.write(f"{i+1:<8} {freq_thz:<20.6f} {fr:<20.6f} {d:<20.10f} {e:<40.10f} {q:<15.2f} {wavelength_nm:<15.2f}\n")
                f.write("\n" + "="*150 + "\n")
                f.write(f"HIGHEST Q MODE:\n")
                f.write("="*150 + "\n")
                f.write(f"  Mode index: {ind+1}\n")
                f.write(f"  Frequency: {freq_maxq:.6f} THz\n")
                f.write(f'  Freq (Meep): {freq_max:.6f}\n')
                f.write(f"  Q factor: {maxq:.2f}\n")
                f.write(f"  Wavelength: {(c/(freq_maxq*1e12))*1e9:.2f} nm\n")
                if ModeVolume:
                    f.write(f"  Mode volume: {(V_mode*length**3)/(wavelength_max_nm/refraction_silicon)**3:.6f} (lambda/n)^3\n")
                    purcell_factor = purcell_prefix * ( wavelength_max_nm/refraction_silicon)**3 * (maxq/ (V_mode * length**3) )
                    f.write(f"  Extrapolated Purcell Factor: {purcell_factor}\n")
                f.write("\n" + "="*150 + "\n")
                f.write(f'QUADRATIC ARRAYS \n')
                f.write(f'Vertical lengths:{verts}\n')
                f.write(f'Horizontal Lengths: {horiz}\n')
                # Save raw data in CSV format for easy importing (Change Later)
                import csv
                with open(os.path.join(dir,"mode_data.csv"), "w", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Mode", "Frequency_THz", 'Freq_Meep' "Q_Factor", "Wavelength_nm"])
                    for i, (freq, q) in enumerate(zip(freqs, Qs)):
                        freq_thz = freq * conv
                        wavelength_nm = (c / (freq_thz * 1e12)) * 1e9
                        writer.writerow([i+1, freq_thz, freq, q, wavelength_nm])
        else:
            print("WARNING: No Good Modes")


    if modevisulization: 
        #Narrow source to excite the mode you want (change frequency according to what mode you want to excite)
        source_new = mp.Source(mp.GaussianSource(freq_max, fwidth = .05), excitation, mp.Vector3(0, 0, 0), size = mp.Vector3(0,0,0))
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
            # Create a list of field components and their names
            field_components = [
                (mp.Ex, "Ex"),
                (mp.Ey, "Ey"),
                (mp.Ez, "Ez"),
                (mp.Hx, "Hx"),
                (mp.Hy, "Hy"),
                (mp.Hz, "Hz")
            ]            
            for field, field_name in field_components:
                f = plt.figure(dpi=150)
                plt.axvline(x = taper_length + .5, color = 'k', linestyle = '--')
                plt.axvline(x = -taper_length -.5 , color = 'k', linestyle = '--')
                sim2.plot2D(ax=f.gca(), fields=field, output_plane = mp.Volume(center = mp.Vector3(0,0,0), size = mp.Vector3(sx + 2 * pml_padding_x,sy + 2 * pml_padding_y,0)))
                plt.title(f'{field_name} Component [xy]')
                plt.xlabel('x (normalized units)')
                plt.ylabel('y (normalized units)')
                plt.savefig(os.path.join(dir, f"{field_name}Componentxy.png"), dpi=300, bbox_inches='tight')
                plt.close()

                f = plt.figure(dpi=150)
                sim2.plot2D(ax=f.gca(), fields=field, output_plane = mp.Volume(center = mp.Vector3(0,0,0), size = mp.Vector3(0,sy + 2 * pml_padding_y,sz + 2 * pml_padding_z)))
                plt.title(f'{field_name} Component [yz]')
                plt.xlabel('y (normalized units)')
                plt.ylabel('z (normalized units)')
                plt.savefig(os.path.join(dir, f"{field_name}Componentyz.png"), dpi=300, bbox_inches='tight')
                plt.close()

                f = plt.figure(dpi=150)
                sim2.plot2D(ax=f.gca(), fields=field, output_plane = mp.Volume(center = mp.Vector3(0,0,0), size = mp.Vector3(sx + 2 * pml_padding_x,0,sz + 2 * pml_padding_z)))
                plt.title(f'{field_name} Component [xz]')
                plt.xlabel('x (normalized units)')
                plt.ylabel('z (normalized units)')
                plt.savefig(os.path.join(dir, f"{field_name}Componentxz.png"), dpi=300, bbox_inches='tight')
                plt.close()

            
    
    if lorentz: 
        # Used for optimization purposes (change freqs[i] accordingly)
        detuning = (freqs[0] - fcenter)/fcenter
        kappa_factors = 2/(maxq)
        transmission = kappa_factors ** 2 / (detuning ** 2 + kappa_factors ** 2)
        return detuning**2
            

    #return Qs, freqs, V_mode, purcell_factor

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
    
