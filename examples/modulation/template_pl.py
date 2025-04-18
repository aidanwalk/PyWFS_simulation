"""
This is a template script for running a modulated PyWFS simulation.
This script is designed to be modified for different modulation schemes or
parameters. It provides a structure for setting up the simulation, running it,
and visualizing the results. The actual implementation of the modulation will
mostly depend on the modulation scheme (i.e. constant radius, random radius,
etc.)

@author : Aidan Walk
created on : Thu Mar 28 11:07 2025

"""

import numpy as np
import hcipy as hp
from astropy.table import Table


import plot_helper_pl as plot_helper


# import simulation modules
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix
from ModulatedPyWFS import ModulatedWavefrontSensor
import aberrations
import stars


# -----------------------------------------------------------------------------
# FUNCTIONS FOR GENERATING THE RESPONSE CURVES
# -----------------------------------------------------------------------------

def average_response(points, WFE, 
                     N_iters=30,
                     WFS=ModulatedWavefrontSensor(),
                     N_stars=12,
                     imat=None):
    """
    Compute the average response of the modulated PyWFS to a given wavefront
    error (WFE) over a specified number of iterations (N_iters).
    This is particualrly useful for averaging out noise in the measurements 
    (when phase is a random power law).

    Parameters
    ----------
    points : _type_
        _description_
    WFE : _type_
        _description_
    """
    outs = []
    for i in range(N_iters):
        print('\t\tIteration', i+1, 'of', N_iters)
        wavefront = WFS.flat_wavefront()
        # Create the input phase to generate the response from
        # WFE in radians for a tilt aberration
        # input_slope = WFE/206265 * WFS.telescope_diameter/2
        # phase = Z.from_name('tilt x', 
        #                     WFE=input_slope, 
        #                     wavelength=WFS.wavelength)
        phase = aberrations.make_noise_pl(WFE, 
                          WFS.pupil.shape[0],
                          WFS.pupil.shape[0], 
                          -9, 
                          WFS.N_elements**2).ravel()
        # Apply the aberration to the wavefront
        wavefront = aberrations.aberrate(wavefront, phase)
        
        wavefront.total_power /= N_stars

        # Modulate the wavefront sensor
        signal = WFS.discrete_modulation(wavefront,
                                         points)
        
        # Measure the slopes of the signal
        sx, sy = WFS.measure_slopes(signal)
        
        # # Compute the mean slope in x. 
        # outs.append(np.mean(sx))
        
        # Compute the RMS of the phase
        p = imat.slope2phase(sx, sy)
        outs.append(np.std(p))
    
    
    return np.mean(outs)




def generate_response(points, WFEs,
                     N_iters=30,
                     WFS=ModulatedWavefrontSensor(),
                     N_stars=12,
                     imat=None):
    out_slope = []
    for i, WFE in enumerate(WFEs):
        print(f'\tWFE ={WFE:0.4f}', f'{i+1}/{len(WFEs)}')
        out_slope.append(average_response(points, WFE,
                                            N_iters=N_iters,
                                            WFS=WFS,
                                            N_stars=N_stars,
                                            imat=imat))
        
    return out_slope
    
    

def make_response_table(modulation_radii=np.linspace(0, 0.4, 4),
                        points_generator=stars.uniform_azimuth,
                        N_stars=12,
                        input_WFE=np.linspace(0, 2*3.14, 4),
                        out_dir='./',
                        out_file='response_curves.txt',
                        WFS=ModulatedWavefrontSensor(),
                        N_iters=30,
                        imat=None):
    
    # Make the response curve for the WFS for each modulation radius
    curves = []
    for i, radius in enumerate(modulation_radii): 
        print('\nModulation radius =', radius, f'{i+1}/{len(modulation_radii)}')
        # Generate an x,y list of modulation points. 
        modulation_points = points_generator(radius, N_points=N_stars)/206265
        out_slopes = generate_response(modulation_points, input_WFE,
                                        N_iters=N_iters,
                                        WFS=WFS,
                                        N_stars=N_stars,
                                        imat=imat)
        
        # Make a table column for this response curve at this radius
        curves.append(out_slopes)
        
    curves = np.array(curves)
    # add the input slopes to a table column 
    col_names = [f'r_{radius:0.2f}' for radius in modulation_radii]
    tab = Table(data=[input_WFE, *curves], names=['input_WFE', *col_names])
    
    tab.write(out_dir+out_file, format='ascii.fixed_width', overwrite=True)
    return


# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# FUNCTIONS FOR VISUALIZING THE MODULATION, RECONSTRUCTION
# -----------------------------------------------------------------------------


def visualize_modulation(radius, 
                         WFS, 
                         N_stars=12,
                         out_dir='./',
                         points_generator=stars.uniform_azimuth,
                         title='Modulation Visualization',):
    # Create a wavefont incoming to the WFS
    incoming_wavefront = WFS.flat_wavefront()
    # phase = Z.from_name('tilt x', WFE=1/206265, wavelength=WFS.wavelength)
    # incoming_wavefront = aberrations.aberrate(incoming_wavefront, phase)

    # x = modulation_radii * np.cos(modulation_thetas)
    # y = modulation_radii * np.sin(modulation_thetas)
    # modulation_points = np.vstack((x, y)).T
    modulation_points = points_generator(radius, N_points=N_stars)/206265
    
    focal_image, pupil_image = WFS.visualize_discrete_modulation(
        incoming_wavefront, 
        modulation_positions=modulation_points
        )
    
    focal_image = hp.Field(focal_image.ravel(), WFS.focal_grid)
    plot_helper.plot_progression(focal_image, pupil_image, 
                                 title=f'{title}, {N_stars} stars',
                                 fname=out_dir+'light_progression.png')
    
    return 


    
def verify_reconstruction(radius,
                          WFE=0.02/206265, 
                          WFS=ModulatedWavefrontSensor(), 
                          Z=None,
                          points_generator=stars.uniform_azimuth,
                          N_stars=12,
                          out_dir='./',
                          title='Reconstruction Verification'):
    
    modulation_points = points_generator(radius, N_points=N_stars) /206265
    
    
    z1 = Z.from_name('tilt x', WFE=WFE*WFS.telescope_diameter/2, wavelength=WFS.wavelength)
    z2 = Z.from_name('spherical', WFE=WFE, wavelength=WFS.wavelength)
    z3 = aberrations.make_noise_pl(2., 
                          WFS.pupil.shape[0],
                          WFS.pupil.shape[0], 
                          -7, 
                          WFS.N_elements**2).ravel()
    z3 = hp.Field(z3, WFS.input_pupil_grid)
    aberrs = [z1, z2, z3]
    
    
    # Create the interaction matrix
    imat = interaction_matrix(WFS.N_elements)

    
    input_phases = []
    output_phases = []
    for i, phase in enumerate(aberrs):
        # Create a wavefont incoming to the WFS
        incoming_wavefront = WFS.flat_wavefront()
        incoming_wavefront = aberrations.aberrate(incoming_wavefront, phase)
        input_phases.append(phase.shaped)
        # Pass the wavefront through the WFS
        signal = WFS.discrete_modulation(incoming_wavefront,
                                         modulation_points)
        
        # Recover the slopes
        sx, sy = WFS.measure_slopes(signal)
        # Use it to solve for phases
        recovered_phase = imat.slope2phase(sx, sy)
        output_phases.append(recovered_phase)
    
    # Make a plot of the recovered phase
    plot_helper.plot_phases(input_phases, output_phases,
                            fname=out_dir+'reconstruction.png', 
                            title=f'{title}, {N_stars} stars, '+\
                                '$r_{mod}$='+f'{radius:.2f} as',)
    return

# -----------------------------------------------------------------------------


    
