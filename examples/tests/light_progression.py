
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
# import Zernike # type: ignore
import aberrations



import hcipy as hp
import matplotlib.pyplot as plt
plt.close('all')  


if __name__ == "__main__":
    # =========================================================================
    # Plot the light progression through the WFS
    # =========================================================================
    N_pupil_px = 2**8
    WFE = np.radians(2/3600)
    
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = WavefrontSensor(pupil_array)
    
    # %%
    # Inject an aberration in to the incoming wavefront
    # Z = Zernike.Zernike(pupil_array, rmax=N_pupil_px/2, wvln=WFS.wavelength)
    Z = aberrations.Zernike(WFS.input_pupil_grid, WFS.telescope_diameter)
    
    # zxm = Z.phase_delay(1,1, WFE=-WFE, wavelength=WFS.wavelength)
    zxm = Z.from_name('tilt x', -WFE, WFS.wavelength)
    zym = Z.from_name('tilt y', WFE=-WFE, wavelength=WFS.wavelength)

    zy = hp.mode_basis.zernike(1,-1, D=WFS.telescope_diameter, grid=WFS.input_pupil_grid)
    zx = hp.mode_basis.zernike(1,1, D=WFS.telescope_diameter, grid=WFS.input_pupil_grid)
    k = 2*np.pi / WFS.wavelength
    zy *= WFE/2 * k
    zx *= WFE/2 * k


    # plt.subplot(121)
    # im = plt.imshow(zym, origin='lower')
    # plt.colorbar(im)

    # plt.subplot(122)
    # im = plt.imshow(zy.shaped, origin='lower')
    # plt.colorbar(im)
    # plt.show()
    # %%
    wavefront = WFS.flat_wavefront()
    wavefront = aberrations.aberrate(wavefront, zxm, zym)
    wavefront.electric_field *= np.exp(1j * (zx + zy))
    progression = WFS.light_progression(wavefront)


    plotter.plot_light_progression(WFS.aperture, *progression, 
                                   fname='light_progression.png')
