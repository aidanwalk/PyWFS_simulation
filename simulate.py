#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 12:18:27 2025

@author: Aidan Walk
"""

import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import plotter
from PyWFS import WaveFrontSensor


path2code = '/home/arcadia/mysoft/gradschool/useful/'
sys.path.append(path2code)
import code_fragments.Wavefront as wf
from code_fragments import Zernike



if __name__ == "__main__":
    N_pupil_px = 2**8
    WFE = np.radians(0.01/3600)
    modulation_radius = 1 # arcsecond
    
    
    # Convert modulation radius to radians
    modulation_radius /= 206265
    
    
    
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = WaveFrontSensor(pupil_array)
    
    # Create a wavefont incoming to the WFS
    incoming_wavefront = WFS.flat_wavefront()
    # Inject an aberration in to the incoming wavefront
    Z = Zernike.Zernike(pupil_array, rmax=N_pupil_px/2, wvln=WFS.wavelength)
    # aberration = Z.Tilt_Y(WFE=WFE, wvln=WFS.wavelength)
    aberration = Z.Tilt_X(WFE=-modulation_radius, wvln=WFS.wavelength)
    # aberration = Z.Spherical(WFE=WFE, wvln=WFS.wavelength)
    # aberration = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -10, 36**2)
    incoming_wavefront.electric_field *= np.exp(1j * aberration.flatten())
    

    # %%
    # Modulate the beam by injecting tip/tilt to steer the PSF around the 
    # pyramid
    signal = WFS.modulate(incoming_wavefront, modulation_radius, num_steps=12)
    
    # Recover the slopes
    sx, sy = WFS.measure_slopes(signal)
    
    
    x, y = WFS.signal_grid.points.T
    x, y = x.reshape(sx.shape), y.reshape(sy.shape)
    plotter.plot_3D_slopes(x, y, sx, sy, './plots/recovered_slopes.html')
    plotter.plot_2D_slopes(sx, sy)
    
    # Save the wavefront slopes as fits files
    hdu = fits.PrimaryHDU(sx)
    hdu.writeto('sx.fits', overwrite=True)
    hdu = fits.PrimaryHDU(sy)
    hdu.writeto('sy.fits', overwrite=True)
    
    
    
    
    
    
    
    #%%
    '''
    # %% In[Use our own PyWFS]
    # =========================================================================
    # Now, lets make our own pyramid wavefront sensor optic, without using the
    # built in HCIpy one. 
    # =========================================================================
    # create a propagator for the light using a telecentric reimaging system
    # Propagator for pupil grid to focal grid
    prop = hp.FraunhoferPropagator(WFS.pupil_grid, WFS.focal_grid)
    # prop = WFS.pupil2image
    # propagator from focal grid to PyWFS grid
    pwfs_grid = hp.make_pupil_grid(2*WFS.Npx_pupil, 2*WFS.telescope_diameter)
    prop_pywfs = hp.FraunhoferPropagator(WFS.focal_grid, pwfs_grid)
    
    
    # Create the incoming wavefront at the telescope aperture
    # wavefront = hp.Wavefront(aperture.flatten(), wvln_wfs)
    # wavefront.total_power = 1
    wavefront = WFS.flat_wavefront()
    
    # Propagate the light to the telescope focal plane
    focal_plane = prop.forward(wavefront)
    
    
    
    # --
    # Trying to follow the code in hcipy pyramid
    # separation = WFS.telescope_diameter
    # refractive_index = 1.5
    # pyramid_surface = -separation / (2 * (refractive_index - 1)) * (np.abs(WFS.focal_grid.x) + np.abs(WFS.focal_grid.y))
    # pyramid1 = hp.optics.SurfaceApodizer(hp.Field(pyramid_surface, WFS.focal_grid), refractive_index)
    # --

    
    import pyramid_optic as pyr
    # @TODO! How to find actual value for phase_delay instead of just guessing?
    pyramid = pyr.pyramid_phase_mask(focal_plane, WFS.Npx_pupil,
                                     phase_delay=6.86e-6, wvln=WFS.wavelength)
    
    # plot the pyramid phase mask
    plotter.plot_pyramid_mask(pyramid)
    
    
    
    focal_plane.electric_field *= np.exp(1j * pyramid.flatten())
    
    # Propagate the focal plane image to the pywfs grid
    pupil = prop_pywfs.forward(focal_plane)
    
    
    plt.figure(2)
    plt.clf()
    plt.title("Focal Plane Image")
    fp_img = np.log10(focal_plane.intensity/focal_plane.intensity.max())
    hp.imshow_field(fp_img, vmin=-6, cmap='bone')
    plt.colorbar()
    plt.show()
    
    hp.imshow_field(pupil.intensity, cmap='bone')
    plt.show()
    '''