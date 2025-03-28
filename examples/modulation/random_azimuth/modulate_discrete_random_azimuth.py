#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Makes a plot of input slopes vs output slopes for a modulated PyWFS as a 
function of modulation radius.

HOWEVER, instead of choosing evenly spaced azimuthal modulation positions (like
in modulate.py), we choose random azimuthal modulation positions. This is to 
test the effect of stars falling randomly on the pyramid array, but all at a 
fixed radius (same gain).


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




def response(WFEs, modulation_points):
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
        signal = WFS.discrete_modulation(incoming_wavefront,
                                         modulation_points)
        
        # Measure the slopes of the signal
        sx, sy = WFS.measure_slopes(signal)
        # Compute the mean slope in x. 
        out_slope.append(np.mean(sx))

    return out_slope




def visualize_modulation(radius):
    global modulation_thetas, file_suffix
    # Create a wavefont incoming to the WFS
    incoming_wavefront = WFS.flat_wavefront()
    # phase = Z.from_name('tilt x', WFE=1/206265, wavelength=WFS.wavelength)
    # incoming_wavefront = aberrations.aberrate(incoming_wavefront, phase)

    x = radius * np.cos(modulation_thetas)
    y = radius * np.sin(modulation_thetas)
    modulation_points = np.vstack((x, y)).T
    
    focal_image, pupil_image = WFS.visualize_discrete_modulation(
        incoming_wavefront, modulation_positions=modulation_points)
    focal_image = hp.Field(focal_image.ravel(), WFS.focal_grid)
    plot_helper.plot_progression(focal_image, pupil_image, 
                                 title='Random Azimuth Sampling - $r_{mod}$='+f'{radius*206265:0.2f} as',
                                 fname=f'light_progression_{file_suffix}.png')




def verify_reconstruction(modulation_radius, WFE=0.02/206265):
    global Z, modulation_thetas, file_suffix
    
    z1 = Z.from_name('tilt x', WFE=WFE*WFS.telescope_diameter/2,
                     wavelength=WFS.wavelength)
    z2 = Z.from_name('spherical', WFE=WFE, wavelength=WFS.wavelength)
    z3 = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -9, WFS.N_elements**2).ravel()
    z3 = hp.Field(z3, WFS.input_pupil_grid)
    # zs = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -10, WFS.N_elements**2)
    aberrs = [z1, z2, z3]
    
    # Create the interaction matrix
    imat = interaction_matrix(WFS.N_elements)

    # Convert modulation positions to x,y coordinates
    x = modulation_radius * np.cos(modulation_thetas)
    y = modulation_radius * np.sin(modulation_thetas)
    modulation_points = np.vstack((x, y)).T


    for i, phase in enumerate(aberrs):
        # Create a wavefont incoming to the WFS
        incoming_wavefront = WFS.flat_wavefront()
        aberrations.aberrate(incoming_wavefront, phase)
        
        # input_phase = incoming_wavefront.phase.shaped
        hdu = fits.PrimaryHDU(phase.shaped)
        hdu.writeto(f'./aberrations/input_{file_suffix}_{i}.fits', overwrite=True)
    
        # Pass the wavefront through the WFS
        signal = WFS.discrete_modulation(incoming_wavefront,
                                         modulation_points)
        
        # Recover the slopes
        sx, sy = WFS.measure_slopes(signal)
        # Use it to solve for phases
        recovered_phase = imat.slope2phase(sx, sy)
        hdu = fits.PrimaryHDU(recovered_phase)
        hdu.writeto(f'./aberrations/recovered_{file_suffix}_{i}.fits', overwrite=True)
    
    
    # Make a plot of the recovered phase
    plot_helper.plot_phases(len(aberrs), prefix=file_suffix, 
                            fname=f'reconstruction_{file_suffix}.png', 
                            title='Random Azimuth Reconstruction, $r_{mod}$='+f'{modulation_radius*206265:0.2f} as')
    return


# %% 

if __name__ == "__main__":
    N_pupil_px = 2**8
    input_slopes = np.linspace(0, 0.2, 21)
    modulation_radii = np.linspace(0, 0.2, 11)
    modulation_steps = 12
    # Suffix to be appended to the output files
    file_suffix = f'random_azimuth_N{modulation_steps}'

    # Randomly generate azimuthal modulation positions
    modulation_thetas = np.random.uniform(0, 2*np.pi, modulation_steps)
    
    
    
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = ModulatedWavefrontSensor(pupil_array)
    
    # -------------------------------------------------------------------------
    # Generate a response curve for each modulation radius
    # -------------------------------------------------------------------------

    # Inject an aberration in to the incoming wavefront
    Z = aberrations.Zernike(WFS.input_pupil_grid, WFS.telescope_diameter)

    curves = []
    for radius in modulation_radii:
        r = np.radians(radius/3600)
        # Convert modulation positions to x,y coordinates
        x = r * np.cos(modulation_thetas)
        y = r * np.sin(modulation_thetas)
        modulation_points = np.vstack((x, y)).T

        print(f'Modulation radius: {radius:0.3f} arcsec')
        output_slopes = response(input_slopes, modulation_points) 
        curves.append(output_slopes)


    # %%
    plot_helper.plot_response(input_slopes, -np.array(curves), modulation_radii,
                              title='PyWFS Gain - Random Azimuth Sampling',
                              fname=f'response_{file_suffix}.png')



    # -------------------------------------------------------------------------
    # Visualize the modulation
    # -------------------------------------------------------------------------
    visualize_modulation(radius=0.4/206265)


    # -------------------------------------------------------------------------
    # Verify wavefront reconstruction
    # -------------------------------------------------------------------------
    # %%
    verify_reconstruction(modulation_radius=0.2/206265, WFE=0.02/206265)

