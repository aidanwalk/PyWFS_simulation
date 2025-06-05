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
from reconstruct import interaction_matrix, zernike_decomposition
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
    N_modes = 10
    WFE = np.radians(0.2/3600)
    
    
    # -------------------------------------------------------------------------
    # Simualte a WFS measurement
    # -------------------------------------------------------------------------
    
    # Init the wavefront sensor
    WFS = WavefrontSensor(pupil=(N_pupil_px, N_pupil_px), N_elements=36)
    
    
    # Inject an aberration in to the incoming wavefront
    Z = aberrations.Zernike(grid=WFS.input_pupil_grid,
                            D=WFS.telescope_diameter)
    # Init the Zernike decomposition object
    Z_decomp = zernike_decomposition(N_modes, grid=WFS.signal_grid,
                                     D=WFS.telescope_diameter)
    
    
    phase = Z.from_name('tilt x', WFE=WFE*WFS.telescope_diameter/2, wavelength=WFS.wavelength)
    # phase += Z.from_name('tilt y', WFE=WFE*WFS.telescope_diameter/2, wavelength=WFS.wavelength)
    phase += Z.from_name('defocus', WFE=0.1/206265, wavelength=WFS.wavelength)
    # pl = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -7, WFS.N_elements**2) * WFS.aperture
    # phase = hp.Field(pl.ravel(), WFS.input_pupil_grid)
    # aberrs = [z1, z2, z3]
    
    # Create the interaction matrix
    imat = interaction_matrix(WFS.N_elements)

    # Create a wavefont incoming to the WFS
    incoming_wavefront = WFS.flat_wavefront()
    aberrations.aberrate(incoming_wavefront, phase)
    
    # Pass the wavefront through the WFS
    signal = WFS.pass_through(incoming_wavefront)
    
    # Recover the slopes
    sx, sy = WFS.measure_slopes(signal)
    # Use it to solve for phases
    recovered_phase = imat.slope2phase(sx, sy) * WFS.circular_aperture((WFS.N_elements, WFS.N_elements), WFS.N_elements/2)
    
    
    # Perform the Zernike decomposition
    coeffs = Z_decomp.decompose(recovered_phase.flatten())
    # Reconstruct the phase just based on the Zernike decomposition
    projection = Z_decomp.project(coeffs).shaped
    
    
    
    plt.figure(figsize=(10,7), tight_layout=True)
    plt.clf()
    plt.subplot(2, 3, 1)
    plt.title('Input Phase')
    plt.imshow(phase.shaped, origin='lower', cmap='bone')
    
    plt.subplot(2, 3, 2)
    plt.title('Recovered Phase')
    plt.imshow(recovered_phase, origin='lower', cmap='bone', vmin=-0.5, vmax=0.5)
    
    plt.subplot(2, 3, 3)
    plt.title('Zernike Projectio, $N_{modes}$=10')
    plt.imshow(projection, origin='lower', cmap='bone', vmin=-0.5, vmax=0.5)
    
    plt.subplot(2, 1, 2)
    plt.title('Zernike Coefficients')
    plt.scatter(np.arange(len(coeffs)), coeffs)
    plt.xlabel('Zernike Mode Index')
    plt.ylabel('Coefficient Value')
    
    
    plt.savefig('modal_phase.png', dpi=300)
    
    

