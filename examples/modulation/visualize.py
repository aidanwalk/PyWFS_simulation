#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize the modulation of a modulated PyWFS. This is a demonstration the 
code actually works. 

Created on Thu Mar 13 12:18:27 2025

@author: Aidan Walk
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from ModulatedPyWFS import ModulatedWavefrontSensor

sys.path.append('/home/arcadia/mysoft/gradschool/useful/')
import code_fragments.Wavefront as wf
from code_fragments import Zernike


def plot_progression(focal_image, pupil_image):
    plt.figure(figsize=(7,4), tight_layout=True)
    plt.suptitle('Modulated Pyramid')

    plt.subplot(121)
    plt.title('focal plane')
    im = plt.imshow(np.log10(focal_image/focal_image.max()), 
                    vmin=-5, vmax=0, cmap='bone', origin='lower')
    plt.axis('off')
    # plt.colorbar(im)

    plt.subplot(122)
    plt.title('pupil plane')
    im = plt.imshow(pupil_image, cmap='bone', origin='lower')
    plt.axis('off')
    # plt.colorbar(im)

    plt.savefig('light_progression.png', dpi=300)
    return 




if __name__ == "__main__":
    N_pupil_px = 2**8
    WFE = np.radians(0/3600)
    modulation_radius = np.radians(1/3600)
    modulation_steps = 12
    
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = ModulatedWavefrontSensor(pupil_array)
    
    
    # Inject an aberration in to the incoming wavefront
    Z = Zernike.Zernike(pupil_array, rmax=N_pupil_px/2, wvln=WFS.wavelength)
    # Create a wavefont incoming to the WFS
    incoming_wavefront = WFS.flat_wavefront()
    aberration = Z.Tilt_X(WFE=WFE, wvln=WFS.wavelength)
    incoming_wavefront.electric_field *= np.exp(1j * aberration.flatten())

    # Modulate the beam by injecting tip/tilt to steer the PSF around the 
    # pyramid
    # signal = WFS.modulate(incoming_wavefront, modulation_radius, num_steps=12)
    focal_image, pupil_image = WFS.visualize_modulation(
        incoming_wavefront, modulation_radius, num_steps=modulation_steps)
    
    plot_progression(focal_image, pupil_image)