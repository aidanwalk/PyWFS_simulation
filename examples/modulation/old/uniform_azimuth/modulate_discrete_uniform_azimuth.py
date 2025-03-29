#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Makes a plot of input slopes vs output slopes for a modulated PyWFS as a 
function of modulation radius.

Created on Thu Mar 13 12:18:27 2025

@author: Aidan Walk
"""

# %%
import sys
import numpy as np
import hcipy as hp
from astropy.io import fits

import plot_helper

import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix
from ModulatedPyWFS import ModulatedWavefrontSensor
import aberrations

sys.path.append('/home/arcadia/mysoft/gradschool/useful/code_fragments/')
import Wavefront as wf




def response(WFEs, modulation_radius):
    out_slope = []
    for WFE in WFEs:
        print(f'\tWFE={WFE:0.3f} arcsec')
        WFE = np.radians(WFE/3600)
        # Create a wavefont incoming to the WFS
        incoming_wavefront = WFS.flat_wavefront()
        aberration = Z.from_name('tilt x', WFE=WFE*WFS.telescope_diameter/2,
                                 wavelength=WFS.wavelength)
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




def visualize_modulation(radius):
    # Create a wavefont incoming to the WFS
    incoming_wavefront = WFS.flat_wavefront()
    # phase = Z.from_name('tilt x', WFE=1/206265, wavelength=WFS.wavelength)
    # incoming_wavefront = aberrations.aberrate(incoming_wavefront, phase)

    # Modulate the beam by injecting tip/tilt to steer the PSF around the 
    # pyramid
    # signal = WFS.modulate(incoming_wavefront, modulation_radius, num_steps=12)
    focal_image, pupil_image = WFS.visualize_modulation(
        incoming_wavefront, radius=radius, num_steps=modulation_steps)
    focal_image = hp.Field(focal_image.ravel(), WFS.focal_grid)
    plot_helper.plot_progression(focal_image, pupil_image, 
                                 title='Uniform Azimuth Sampling - $r_{mod}$='+f'{radius*206265:0.2f} as',
                                 fname='light_progression_uniform_azimuth.png')
    
    return




def verify_reconstruction(modulation_radius, WFE=0.02/206265):
    global Z, modulation_steps
    file_prefix= 'modulated_uniform_azimuth'
    
    z1 = Z.from_name('tilt x', WFE=WFE*WFS.telescope_diameter/2,
                     wavelength=WFS.wavelength)
    z2 = Z.from_name('spherical', WFE=WFE, wavelength=WFS.wavelength)
    z3 = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -9, WFS.N_elements**2).ravel()
    z3 = hp.Field(z3, WFS.input_pupil_grid)
    # zs = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -10, WFS.N_elements**2)
    aberrs = [z1, z2, z3]
    
    # Create the interaction matrix
    imat = interaction_matrix(WFS.N_elements)
    
    for i, phase in enumerate(aberrs):
        # Create a wavefont incoming to the WFS
        incoming_wavefront = WFS.flat_wavefront()
        aberrations.aberrate(incoming_wavefront, phase)
        
        # input_phase = incoming_wavefront.phase.shaped
        hdu = fits.PrimaryHDU(phase.shaped)
        hdu.writeto(f'./aberrations/input_{file_prefix}_{i}.fits', overwrite=True)
    
        # Pass the wavefront through the WFS
        signal = WFS.modulate(incoming_wavefront, 
                              radius=modulation_radius, 
                              num_steps=modulation_steps
                              )
        
        # Recover the slopes
        sx, sy = WFS.measure_slopes(signal)
        # Use it to solve for phases
        recovered_phase = imat.slope2phase(sx, sy)
        hdu = fits.PrimaryHDU(recovered_phase)
        hdu.writeto(f'./aberrations/recovered_{file_prefix}_{i}.fits', overwrite=True)
    
    
    # Make a plot of the recovered phase
    plot_helper.plot_phases(len(aberrs), prefix=file_prefix, 
                            fname='reconstruction_uniform_azimuth.png', 
                            title='Uniform Azimuth Reconstruction, $r_{mod}$='+f'{modulation_radius*206265:0.2f} as')
    return




# %%

if __name__ == "__main__":
    N_pupil_px = 2**8
    input_slopes = np.linspace(0, 0.2, 21) 
    modulation_radii = np.linspace(0, 0.2, 11)
    modulation_steps = 12
    
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = ModulatedWavefrontSensor(pupil_array)
    
    
    # Inject an aberration in to the incoming wavefront
    Z = aberrations.Zernike(WFS.input_pupil_grid, WFS.telescope_diameter)

    # %%
    # -------------------------------------------------------------------------
    # Generate a response curve for each modulation radius
    # -------------------------------------------------------------------------
    curves = []
    for radius in modulation_radii:
        print(f'Modulation radius: {radius:0.3f} arcsec')
        output_slopes = response(input_slopes, radius) 
        curves.append(output_slopes)
    # %%
    
    plot_helper.plot_response(input_slopes, np.abs(curves), modulation_radii,
                              title='PyWFS Gain - Uniform Azimuth Sampling',
                              fname='response_uniform_azimuth.png')


    # %%
    # -------------------------------------------------------------------------
    # Visualize the modulation
    # -------------------------------------------------------------------------
    visualize_modulation(radius=0.4/206265)
    
    # -------------------------------------------------------------------------
    # Verify wavefront reconstruction
    # -------------------------------------------------------------------------
    # %%
    verify_reconstruction(modulation_radius=0.4/206265)