#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Makes a plot of input slopes vs output slopes for a modulated PyWFS as a 
function of modulation radius.

Created on Thu Mar 13 12:18:27 2025

@author: Aidan Walk
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.close('all')

import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from ModulatedPyWFS import ModulatedWavefrontSensor

sys.path.append('/home/arcadia/mysoft/gradschool/useful/code_fragments/')
import Wavefront as wf
import Zernike




def response(WFEs, modulation_radius):
    out_slope = []
    for WFE in WFEs:
        print(f'\tWFE={WFE:0.3f} arcsec')
        WFE = np.radians(WFE/3600)
        # Create a wavefont incoming to the WFS
        incoming_wavefront = WFS.flat_wavefront()
        aberration = Z.Tilt_X(WFE=WFE, wvln=WFS.wavelength)
        incoming_wavefront.electric_field *= np.exp(1j * aberration.flatten())
        incoming_wavefront.total_power /= modulation_steps

        # Modulate the wavefront sensor
        signal = WFS.modulate(incoming_wavefront, 
                              radius=np.radians(modulation_radius/3600), 
                              num_steps=modulation_steps)
        # Measure the slopes of the signal
        sx, sy = WFS.measure_slopes(signal)
        # Compute the mean slope in x. 
        out_slope.append(np.mean(sx))

    return out_slope




if __name__ == "__main__":
    N_pupil_px = 2**8
    input_slopes = np.linspace(0, 0.1, 21)
    modulation_radii = np.linspace(0, 0.1, 11)
    modulation_steps = 12
    
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = ModulatedWavefrontSensor(pupil_array)
    
    
    # Inject an aberration in to the incoming wavefront
    Z = Zernike.Zernike(pupil_array, rmax=N_pupil_px/2, wvln=WFS.wavelength)

    curves = []
    for radius in modulation_radii:
        print(f'Modulation radius: {radius:0.3f} arcsec')
        output_slopes = response(input_slopes, radius) 
        curves.append(output_slopes)


    # %%
    
    plt.figure(figsize=(6,4), tight_layout=True)
    plt.clf()

    colors = cm.inferno(np.linspace(0, 1, len(curves)+1))
    for i, curve in enumerate(curves):
        plt.plot(input_slopes, curve, color=colors[i],
                 label=f'{modulation_radii[i]:0.2f} as')
    
    plt.legend()
    plt.axis('equal')
    plt.title('Modulation Radius')
    plt.xlabel('input slopes [arcsec]')
    plt.ylabel('output slopes [arcsec]')
    plt.savefig('response.png', dpi=300)
