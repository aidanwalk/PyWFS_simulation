
"""
This example shows how to plot the light progression through the WFS. 

Created on Mon Mar 17 14:43 2025

@author: Aidan Walk
"""

import numpy as np
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix
from PyWFS import WavefrontSensor
import plotter
import aberrations


path2code = '/home/arcadia/mysoft/gradschool/useful/code_fragments/'
sys.path.append(path2code)
import Wavefront as wf # type: ignore


import matplotlib.pyplot as plt
plt.close('all')  


if __name__ == "__main__":
    # =========================================================================
    # Plot the light progression through the WFS
    # =========================================================================
    N_pupil_px = 2**8
    WFE = np.radians(0.5/3600)
    
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = WavefrontSensor(pupil_array)
    
    # %%
    # Inject an aberration in to the incoming wavefront
    Z = aberrations.Zernike(WFS.input_pupil_grid, WFS.telescope_diameter)
    
    phase = Z.from_name('tilt x', WFE=WFE, wavelength=WFS.wavelength)


    # %%
    wavefront = WFS.flat_wavefront()
    wavefront = aberrations.aberrate(wavefront, phase)
    progression = WFS.light_progression(wavefront)


    plotter.plot_light_progression(WFS.aperture, *progression, 
                                   fname='light_progression.png')
