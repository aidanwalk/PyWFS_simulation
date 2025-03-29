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

sys.path.append('/home/arcadia/mysoft/gradschool/useful/code_fragments/')
import Wavefront as wf


# -----------------------------------------------------------------------------
# FUNCTIONS TO GENERATE MODULATION POINTS
# -----------------------------------------------------------------------------

def uniform_azimuth(radius, N_points):
    # Generate discrete points in a circle at which to steer the wavefront
    theta = np.linspace(0, 2*np.pi, N_points, endpoint=False)
    x_modulation = radius * np.cos(theta)
    y_modulation = radius * np.sin(theta)
    
    modulation_positions = np.vstack((x_modulation, y_modulation)).T
    return modulation_positions


def random_azimuth(radius, N_points):
    # Generate random azimuthal angles for the modulation points
    theta = np.random.uniform(0, 2*np.pi, N_points)
    x_modulation = radius * np.cos(theta)
    y_modulation = radius * np.sin(theta)
    
    modulation_positions = np.vstack((x_modulation, y_modulation)).T
    return modulation_positions


def random_radius(radius, N_points):
    # Generate points randomly on the grid 
    return np.random.uniform(-radius, radius, (N_points,2))

# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# FUNCTIONS FOR GENERATING THE RESPONSE CURVES
# -----------------------------------------------------------------------------

def average_response(points, WFE):
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
        phase = wf.make_noise_pl(WFE, 
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




def generate_response(points, WFEs):
    out_slope = []
    for i, WFE in enumerate(WFEs):
        print(f'\tWFE ={WFE:0.4f}', f'{i+1}/{len(WFEs)}')
        out_slope.append(average_response(points, WFE))
        
    return out_slope
    
    

def make_response_table():
    # Make the response curve for the WFS for each modulation radius
    curves = []
    for i, radius in enumerate(modulation_radii): 
        print('\nModulation radius =', radius, f'{i+1}/{len(modulation_radii)}')
        # Generate an x,y list of modulation points. 
        modulation_points = points_generator(radius, N_points=N_stars)/206265
        out_slopes = generate_response(modulation_points, input_WFE)
        
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
    
    
    
def verify_reconstruction(radius, WFE=0.02/206265, 
                          title='Reconstruction Verification'):
    modulation_points = points_generator(radius, N_points=N_stars) /206265
    
    z1 = Z.from_name('tilt x', WFE=WFE*WFS.telescope_diameter/2, 
                     wavelength=WFS.wavelength)
    z2 = Z.from_name('spherical', WFE=WFE, wavelength=WFS.wavelength)
    z3 = wf.make_noise_pl(2., 
                          WFS.pupil.shape[0],
                          WFS.pupil.shape[0], 
                          -7, 
                          WFS.N_elements**2).ravel()
    
    z3 = hp.Field(z3, WFS.input_pupil_grid)
    # zs = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -10, WFS.N_elements**2)
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
                            title=f'Random Radius Reconstruction, {N_stars} stars, '+\
                                '$r_{mod}$='+f'{radius:.2f} as',)
    return

# -----------------------------------------------------------------------------



if __name__ == '__main__':
    """
    Simulate the response of a modulated PyWFS to a range of modulation 
    radii. 

    Parameters
    ----------
    out_file : str, optional
        file name for the output response cuves,
        by default 'response_curves.txt'
    points_generator : callable, optional
        function used to generate the modulation points,
        by default uniform_azimuth. Can use any uniform_azimuth,
        random_azimuth, or random_radius (defined above).
    N_stars : int, optional
        The number of stars (modulation points), by default 12
    modulation_radii : list, optional
        radii to modulate over, by default np.linspace(0, 1, 6)
    input_WFE : _type_, optional
        _description_, by default np.linspace(0, 0.5, 6)
    WFS_kwargs : dict, optional
        _description_, by default {}
    N_iters : int, optional
        _description_, by default 1
    quantifier : str, optional
        _description_, by default 'mean slope'
    """
    
    # GLOBAL PARAMETERS
    # -----------------------------
    plot_title = 'Random Radius Sampling'
    points_generator=random_radius
    out_file='response_curves.txt' 
    out_dir = './'
    N_stars=12
    modulation_radii=np.linspace(0, 0.4, 11)
    input_WFE = np.linspace(0, 2*3.14, 11)
    WFS_kwargs={}
    N_iters=10
    
    
    # Init the wavefront sensor and the interaction matrix
    WFS = ModulatedWavefrontSensor(**WFS_kwargs)
    imat = interaction_matrix(WFS.N_elements)
    Z = aberrations.Zernike(WFS.input_pupil_grid, WFS.telescope_diameter)
    
    
    
    visualize_modulation(radius=0.4, 
                         title=plot_title)
    
    verify_reconstruction(radius=0.4, 
                          WFE=0.1/206265, 
                          title=plot_title)
    
    make_response_table()
    
    plot_helper.plot_response(data_file=out_dir+out_file, 
                              title=plot_title, 
                              fname=out_dir+'response.png')

    
    
    
    
