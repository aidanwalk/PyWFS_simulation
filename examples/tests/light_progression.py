
"""
This example shows how to plot the light progression through the WFS. 

Created on Mon Mar 17 14:43 2025

@author: Aidan Walk
"""

import numpy as np
import hcipy as hp
import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix
from PyWFS import WavefrontSensor
import plotter
import aberrations
# 


import matplotlib.pyplot as plt
plt.close('all')  


if __name__ == "__main__":
    # =========================================================================
    # Plot the light progression through the WFS
    # =========================================================================
    N_pupil_px = 2**8
    WFE = np.radians(1/3600)
    
    # Init the wavefront sensor
    WFS = WavefrontSensor()
    
    # Inject an aberration in to the incoming wavefront
    Z = aberrations.Zernike(WFS.input_pupil_grid, WFS.telescope_diameter)
    phase = Z.from_name('tilt x', WFE=WFE*WFS.telescope_diameter/2,
                        wavelength=WFS.wavelength)
    phase += Z.from_name('tilt y', WFE=WFE*WFS.telescope_diameter/2,
                        wavelength=WFS.wavelength)
    # phase = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -4, WFS.N_elements**2).ravel()

    # Initialize the wavefront
    wavefront = WFS.flat_wavefront()
    # Apply the aberration to the wavefront
    wavefront = aberrations.aberrate(wavefront, phase)
    # Propagate the wavefront to the WFS
    aberration, focal_plane, pyramid, WFS_signal = WFS.light_progression(wavefront)
    
    
    
    # Make a plot of the light progression through the WFS
    fig, ax = plt.subplots(nrows=1, ncols=4, 
                           tight_layout=True, 
                           figsize=(13,3))
    plt.suptitle('WFS Light Progression')
    
    # First, plot the incoming wavefront aberration
    ax[0].set_title('Incoming Wavefront Phase')
    im = ax[0].imshow(phase.shaped, cmap='hsv', origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Phase [rad]')
    # Overplot the aperture of the telescope
    # alpha = ~WFS.aperture
    # ax[0].imshow(~WFS.aperture, alpha=alpha.astype(float), cmap='Greys')
    ax[0].axis('off')
    
    
    ax[1].set_title('Focal Plane PSF')
    img = np.log10(focal_plane / focal_plane.max())
    img = hp.Field(img.ravel(), WFS.focal_grid)
    plt.subplot(142)
    im = hp.imshow_field(img, cmap='bone', vmin=-6, vmax=0, grid_units=1/206265, origin='lower') # type: ignore
    # im = ax[1].imshow(img, cmap='bone', vmin=-6, vmax=0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    
    ax[2].set_title('Pyramid Phase Mask')
    im = ax[2].imshow(pyramid, cmap='hsv', origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Phase [rad]')
    ax[2].axis('off')
    
    ax[3].set_title('WFS Signal')
    im = ax[3].imshow(WFS_signal, cmap='bone', origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax[3].axis('off')
    
    plt.savefig('light_progression.png', dpi=300)
