#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Makes a plot of input slopes vs output slopes for a modulated PyWFS as a 
function of modulation radius.

The "input slope" is the rms wavefront deviation in radians given by a power
law wavefront aberration. The "output slope" is the measured rms of the 
recovered wavefront. 


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
    # Create the interaction matrix
    imat = interaction_matrix(WFS.N_elements)
    # Since the power law phase screen is generated randomly, average over many
    # realizations to get a good estimate of the response curve.
    def average_response():
        n, o = [], []
        for i in range(50):
            # WFE = np.radians(WFE/3600)
            # Create a wavefont incoming to the WFS
            incoming_wavefront = WFS.flat_wavefront()
            # Create the power law phase screen coming into the telescope aperture.
            aberration = wf.make_noise_pl(WFE, N_pupil_px, N_pupil_px, -9, WFS.N_elements**2).ravel()
            n.append(np.std(aberration))
            
            
            incoming_wavefront = aberrations.aberrate(incoming_wavefront, 
                                                      aberration
                                                      )
            incoming_wavefront.total_power /= modulation_steps

            # Modulate the wavefront sensor
            signal = WFS.modulate(incoming_wavefront, 
                                radius=np.radians(modulation_radius/3600), 
                                num_steps=modulation_steps)
            
            # Measure the slopes of the signal
            sx, sy = WFS.measure_slopes(signal)
            recovered_phase = imat.slope2phase(sx, sy)
            # Compute the mean slope in x. 
            o.append(np.std(recovered_phase))
        return np.mean(n), np.mean(o)
        
    
    
    out_slope = []
    in_slope = []
    
    for WFE in WFEs:
        inn, out = average_response()
        print(f'\tWFE={WFE:0.3f} radians')
        in_slope.append(inn)
        out_slope.append(out)
        

    return in_slope, out_slope




def visualize_modulation(radius, fname='plot.png'):
    # Create a wavefont incoming to the WFS
    incoming_wavefront = WFS.flat_wavefront()
    # phase = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -9, WFS.N_elements**2).ravel()
    # incoming_wavefront = aberrations.aberrate(incoming_wavefront, phase)

    # Modulate the beam by injecting tip/tilt to steer the PSF around the 
    # pyramid
    # signal = WFS.modulate(incoming_wavefront, modulation_radius, num_steps=12)
    focal_image, pupil_image = WFS.visualize_modulation(
        incoming_wavefront, radius=radius, num_steps=modulation_steps)
    focal_image = hp.Field(focal_image.ravel(), WFS.focal_grid)
    
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7,4.2), tight_layout=True)
    plt.suptitle('Uniform Azimuth Sampling - $r_{mod}$='+f'{radius*206265:0.2f} as')

    plt.subplot(121)
    plt.title('focal plane')
    # im = plt.imshow(np.log10(focal_image/focal_image.max()), 
    #                 vmin=-5, vmax=0, cmap='bone', origin='lower')
    hp.imshow_field(np.log10(focal_image/focal_image.max()),
                    grid_units=1/206265, # 
                    vmin=-5, vmax=0, cmap='bone', origin='lower')
    plt.xlabel('x [arcsec]')
    plt.ylabel('y [arcsec]')
    # plt.axis('off')
    # plt.colorbar(im)

    plt.subplot(122)
    plt.title('pupil plane')
    im = plt.imshow(pupil_image, cmap='bone', origin='lower')
    plt.axis('off')
    # plt.colorbar(im)
    plt.savefig(fname, dpi=300)
    # -------------------------------------------------------------------------
    
    return




def verify_reconstruction(modulation_radius, WFE=0.02/206265, fname='plot.png'):
    global Z, modulation_steps
    z1 = Z.from_name('tilt x', WFE=WFE, wavelength=WFS.wavelength)
    z2 = Z.from_name('spherical', WFE=WFE, wavelength=WFS.wavelength)
    z3 = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -9, WFS.N_elements**2).ravel()
    z3 = hp.Field(z3, WFS.input_pupil_grid)
    # zs = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -10, WFS.N_elements**2)
    aberrs = [z1, z2, z3]
    
    # Create the interaction matrix
    imat = interaction_matrix(WFS.N_elements)
    
    out_phases = []
    for i, phase in enumerate(aberrs):
        # Create a wavefont incoming to the WFS
        incoming_wavefront = WFS.flat_wavefront()
        aberrations.aberrate(incoming_wavefront, phase)
    
        # Pass the wavefront through the WFS
        signal = WFS.modulate(incoming_wavefront, 
                              radius=modulation_radius, 
                              num_steps=modulation_steps
                              )
        
        # Recover the slopes
        sx, sy = WFS.measure_slopes(signal)
        # Use it to solve for phases
        recovered_phase = imat.slope2phase(sx, sy)
        out_phases.append(recovered_phase)
    
    
    
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    
    pltkwargs = {'origin':'lower',
                    'cmap':'bone',
                    }
    
    plt.suptitle('Uniform Azimuth Reconstruction, $r_{mod}$='+f'{modulation_radius*206265:0.2f} as')
    for i in range(len(aberrs)):
        # Create a plot for the x-slope
        im = axs[0,i].imshow(aberrs[i].shaped, **pltkwargs)
        plt.colorbar(im,fraction=0.046, pad=0.04)
        axs[0,i].axis('off')
        im = axs[1,i].imshow(out_phases[i], vmin=-out_phases[i].max(), 
                            vmax=out_phases[i].max(), 
                            **pltkwargs
                            )
        plt.colorbar(im,fraction=0.046, pad=0.04)
        axs[1,i].axis('off')
        
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    # -------------------------------------------------------------------------
    
    return




# %%

if __name__ == "__main__":
    N_pupil_px = 2**8
    input_slopes = np.linspace(0, 2*3.14, 21)
    modulation_radii = np.linspace(0, 0.2, 11)
    modulation_steps = 12
    
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = ModulatedWavefrontSensor(pupil_array)
    
    
    # Inject an aberration in to the incoming wavefront
    Z = aberrations.Zernike(WFS.input_pupil_grid, WFS.telescope_diameter)


    # -------------------------------------------------------------------------
    # Generate a response curve for each modulation radius
    # -------------------------------------------------------------------------
    curves = []
    input_slope_act = []
    for radius in modulation_radii:
        print(f'Modulation radius: {radius:0.3f} arcsec')
        input_slope, output_slopes = response(input_slopes, radius) 
        curves.append(output_slopes)
        input_slope_act.append(input_slope)
    # %%
    import matplotlib.pyplot as plt
    plt.close('all')
    plot_helper.plot_response(input_slope_act, curves, modulation_radii,
                              title='PyWFS Gain - Uniform Azimuth Sampling',
                              fname='response_uniform_azimuth_pl.png')



    # -------------------------------------------------------------------------
    # Visualize the modulation
    # -------------------------------------------------------------------------
    visualize_modulation(radius=0.4/206265, 
                         fname='light_progression_uniform_azimuth_pl.png')
    
    # -------------------------------------------------------------------------
    # Verify wavefront reconstruction
    # -------------------------------------------------------------------------
    # %%
    verify_reconstruction(modulation_radius=0.2/206265, 
                          fname='reconstruction_uniform_azimuth_pl.png')