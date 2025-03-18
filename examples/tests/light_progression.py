
"""
This example shows how to plot the light progression through the WFS. 

Created on Mon Mar 17 14:43 2025

@author: Aidan Walk
"""

import numpy as np
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix
from PyWFS import WaveFrontSensor
import plotter

path2code = '/home/arcadia/mysoft/gradschool/useful/'
sys.path.append(path2code)
import code_fragments.Wavefront as wf
from code_fragments import Zernike


if __name__ == "__main__":
    # =========================================================================
    # Plot the light progression through the WFS
    # =========================================================================
    N_pupil_px = 2**8
    WFE = np.radians(1/3600)
    
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = WaveFrontSensor(pupil_array)
    
    
    # Inject an aberration in to the incoming wavefront
    Z = Zernike.Zernike(pupil_array, rmax=N_pupil_px/2, wvln=WFS.wavelength)
    
    zx = Z.Tilt_X(WFE=WFE, wvln=WFS.wavelength)
    zy = Z.Tilt_Y(WFE=WFE, wvln=WFS.wavelength)

    wavefront = WFS.flat_wavefront()
    wavefront.electric_field *= np.exp(1j * (zx.flatten() + zy.flatten()))
    progression = WFS.light_progression(wavefront)


    plotter.plot_light_progression(WFS.aperture, *progression, 
                                   fname='light_progression.png')
