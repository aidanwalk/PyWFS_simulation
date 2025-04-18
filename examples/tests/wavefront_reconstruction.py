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
import hcipy as hp
from astropy.io import fits
import matplotlib.pyplot as plt


path2code = '/home/arcadia/mysoft/gradschool/useful/code_fragments/'
sys.path.append(path2code)
import Wavefront as wf # type: ignore

import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix
from PyWFS import WavefrontSensor
import aberrations



def plot_phases(fname='phase_comparison.html'):
    
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    plt.suptitle('Unmodulated PyWFS, Input vs Recovered Phase')
    for i in range(len(aberrs)):
        input_phase = fits.getdata(f'./aberrations/input_{i}.fits')*WFS.aperture
        recovered_phase = fits.getdata(f'./aberrations/recovered_{i}.fits')
        # Create a plot for the x-slope
        pltkwargs = {'origin':'lower',
                     'cmap':'bone',
                     }
        im = axs[0,i].imshow(input_phase, **pltkwargs)
        plt.colorbar(im,fraction=0.046, pad=0.04)
        axs[0,i].axis('off')
        im = axs[1,i].imshow(recovered_phase, **pltkwargs)
        plt.colorbar(im,fraction=0.046, pad=0.04)
        axs[1,i].axis('off')
    plt.tight_layout()
    plt.savefig('wavefront_reconstruction.png', dpi=300)
    
    return



if __name__ == "__main__":
    N_pupil_px = 2**8
    WFE = np.radians(0.1/3600)
    
    
    # -------------------------------------------------------------------------
    # Simualte a WFS measurement
    # -------------------------------------------------------------------------
    
    # Init the wavefront sensor
    WFS = WavefrontSensor()
    
    
    # Inject an aberration in to the incoming wavefront
    Z = aberrations.Zernike(grid=WFS.input_pupil_grid,
                            D=WFS.telescope_diameter)
    
    
    z1 = Z.from_name('tilt x', WFE=WFE*WFS.telescope_diameter/2, wavelength=WFS.wavelength)
    # z1 += Z.from_name('tilt y', WFE=WFE*WFS.telescope_diameter/2, wavelength=WFS.wavelength)
    z2 = Z.from_name('spherical', WFE=WFE, wavelength=WFS.wavelength)
    z3 = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -10, WFS.N_elements**2).ravel()
    z3 = hp.Field(z3, WFS.input_pupil_grid)
    aberrs = [z1, z2, z3]
    
    # Create the interaction matrix
    imat = interaction_matrix(WFS.N_elements)

    for i, phase in enumerate(aberrs):
        # Create a wavefont incoming to the WFS
        incoming_wavefront = WFS.flat_wavefront()
        aberrations.aberrate(incoming_wavefront, phase)
        
        # input_phase = incoming_wavefront.phase.shaped
        hdu = fits.PrimaryHDU(phase.shaped)
        hdu.writeto(f'./aberrations/input_{i}.fits', overwrite=True)
    
        # Pass the wavefront through the WFS
        signal = WFS.pass_through(incoming_wavefront)
        
        # Recover the slopes
        sx, sy = WFS.measure_slopes(signal)
        # Use it to solve for phases
        recovered_phase = imat.slope2phase(sx, sy)
        hdu = fits.PrimaryHDU(recovered_phase)
        hdu.writeto(f'./aberrations/recovered_{i}.fits', overwrite=True)
    
    
    # Make a plot of the recovered phase
    plot_phases()
    

