#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests the entire simulation -- do we reconstruct the wavefront we put into the 
simulation?

Created on Fri Mar 14 14:58:43 2025

@author: Aidan Walk
"""


import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt



path2code = '/home/arcadia/mysoft/gradschool/useful/'
sys.path.append(path2code)
import code_fragments.Wavefront as wf
from code_fragments import Zernike

import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/PyWFS_simulation/')
from reconstruct import interaction_matrix
from PyWFS import WaveFrontSensor

def plot_phases(fname='phase_comparison.html'):
    
    fig, axs = plt.subplots(2, 3)
    plt.suptitle('Input vs Recovered Phase')
    for i in range(len(aberrs)):
        input_phase = fits.getdata(f'input_{i}.fits')*WFS.aperture
        recovered_phase = fits.getdata(f'recovered_{i}.fits')
        # Create a plot for the x-slope
        im = axs[0,i].imshow(input_phase, origin='lower')
        # plt.colorbar(im)
        im = axs[1,i].imshow(recovered_phase, origin='lower')
        # plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('wavefront_reconstruction.png', dpi=300)
       
    
    
    
    
    return

if __name__ == "__main__":
    N_pupil_px = 2**8
    WFE = np.radians(0.01/3600)
    modulation_radius = 1 # arcsecond
    
    
    # -------------------------------------------------------------------------
    # Simualte a WFS measurement
    # -------------------------------------------------------------------------
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = WaveFrontSensor(pupil_array)
    
    
    # Inject an aberration in to the incoming wavefront
    Z = Zernike.Zernike(pupil_array, rmax=N_pupil_px/2, wvln=WFS.wavelength)
    
    
    zx = Z.Tilt_X(WFE=WFE, wvln=WFS.wavelength)
    zy = Z.Tilt_Y(WFE=WFE, wvln=WFS.wavelength)
    zs = Z.Spherical(WFE=WFE, wvln=WFS.wavelength)
    # zs = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -10, WFS.N_elements**2)
    aberrs = [zx, zy, zs]
    
    # Create the interaction matrix
    imat = interaction_matrix(WFS.N_elements)

    for i, ab in enumerate(aberrs):
        # Create a wavefont incoming to the WFS
        incoming_wavefront = WFS.flat_wavefront()
        incoming_wavefront.electric_field *= np.exp(1j * ab.flatten())
        
        input_phase = incoming_wavefront.phase.shaped
        hdu = fits.PrimaryHDU(input_phase)
        hdu.writeto(f'input_{i}.fits', overwrite=True)
    
        # Pass the wavefront through the WFS
        signal = WFS.pass_through(incoming_wavefront)
        
        # Recover the slopes
        sx, sy = WFS.measure_slopes(signal)
        # Use it to solve for phases
        recovered_phase = imat.slope2phase(sx, sy)
        hdu = fits.PrimaryHDU(recovered_phase)
        hdu.writeto(f'recovered_{i}.fits', overwrite=True)
    
    
    # Make a plot of the recovered phase
    plot_phases()
