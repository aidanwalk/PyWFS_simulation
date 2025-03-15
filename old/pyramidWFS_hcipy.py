#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:09:55 2024

@author: Aidan Walk
"""

import numpy as np
import hcipy as hp
import copy
import matplotlib.pyplot as plt
import mpld3
from astropy.io import fits
plt.close('all')

import sys
path2code = '/home/arcadia/mysoft/gradschool/useful/'
sys.path.append(path2code)
import code_fragments.Wavefront as wf
from code_fragments import Zernike
from code_fragments import Coordinates as coords


import pyramid_optic as pyr
import plotter

# Wavefront sensor and telescope parameters
wvln_wfs = 800e-9 # meters
telescope_dia = 2.2 # meters
NPIX_FOC = 500
focal_extent = np.radians(5/3600)
# WFE = np.radians(1.25/3600)
WFE = np.radians(0.01/3600)

# Pupil sampling
N_pupil_px = 2**8
pupil_sampling_WFS = 36
N_pupil_px_WFS = 36


def quadrant_mask(pts):
    # Facets of the pyramid
    Q1 = (pts[0]>=0) & (pts[1]>0)
    Q2 = (pts[0]<0) & (pts[1]>=0)
    Q3 = (pts[0]<=0) & (pts[1]<0)
    Q4 = (pts[0]>0) & (pts[1]<=0)
    
    return [Q1, Q2, Q3, Q4]



def find_slopes(I):
    # Compute the mean intensity per pixel
    I0 = (I[0,0]+I[0,1]+I[1,0]+I[1,1])/4
    
    # Compute the WFS slopes based on a quad-cell between the four pupil images
    sx = ( (I[0,1]+I[0,0]) - (I[1,1]+I[1,0]) ) / I0
    sy = ( (I[0,1]+I[1,1]) - (I[0,0]+I[1,0]) ) / I0
    return sx, sy


if __name__ == "__main__":
    
    ...
    # %% In[Create Grids]
    # =========================================================================
    # CREATE HCIPY GRIDS
    # =========================================================================
    # Create a grid that defines the input wavefront
    
    # Make the pupil grid (incoming pupil of the telescope)
    pupil_grid = hp.make_pupil_grid(N_pupil_px, telescope_dia)
    
    # Make the focal grid (This is where the pyramid array will be)
    focal_grid = hp.make_uniform_grid([NPIX_FOC, NPIX_FOC], 
                                      [focal_extent, focal_extent])
    
    # Make the grid for the pwfs (grid the pupil images are projected on to)
    pwfs_grid = hp.make_pupil_grid(2*N_pupil_px_WFS, 2*telescope_dia)
    
    
    
    # %% In[Initialize Wavefront]
    # =========================================================================
    # INIT THE INCOMING WAVEFRONT
    # =========================================================================
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    aperture = hp.Field(pupil_array, pupil_grid)
    
    
    
    
    # %% In[Focal Plane Image]
    # create a propagator for the light using a telecentric reimaging system
    # Propagator for pupil grid to focal grid
    prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)
    
    # Create the incoming wavefront at the telescope aperture
    wavefront = hp.Wavefront(aperture.flatten(), wvln_wfs)
    wavefront.total_power = 1
    
    # Add an aberration to the incoming wavefront
    Z = Zernike.Zernike(pupil_array, rmax=N_pupil_px/2, wvln=wvln_wfs)
    aberration = Z.Tilt_X(WFE=WFE, wvln=wvln_wfs)
    wavefront.electric_field *= np.exp(1j * aberration.flatten())
    # aberration = Z.Tilt_Y(WFE=WFE, wvln=wvln_wfs)
    # aberration = Z.Defocus(WFE=WFE, wvln=wvln_wfs)
    # aberration = Z.Spherical(WFE=WFE, wvln=wvln_wfs)
    # wavefront.electric_field *= np.exp(1j * aberration.flatten())
    
    # aberration = wf.make_noise_pl(2, N_pupil_px, N_pupil_px, -10, 36**2)
    # wavefront.electric_field *= np.exp(1j * aberration.flatten())
    
    incoming_wavefront = copy.deepcopy(wavefront)
    
    
    plotter.plot_wavefront_phase(wavefront.phase.shaped * aperture)
    x = wavefront.phase.shaped.shape[0]
    x = range(x)
    x, y = np.meshgrid(x, x)
    plotter.plot_phase(x, y, np.array(wavefront.phase.shaped.data), fname='actual_phase.html')
    # Equivalently, I could also pass 'aberration' into this plotting function, 
    # except this is total WFE in radians (not WFE%pi like is returned from 
    # 'wavefront.phase')
    
    
    # Propagate the light to the telescope focal plane
    focal_plane = prop.forward(wavefront)
    
    # Plot the focal plane
    plotter.plot_focal_plane(focal_plane)
    
    
    
    
    # %% In[Use HCIpy PyWFS]
    # =========================================================================
    # Propogate the incoming wavefront via hcipy 
    # PyramidWavefrontSensorOptics
    # =========================================================================
    pwfs = hp.PyramidWavefrontSensorOptics(pwfs_grid, 
                                           separation=telescope_dia, 
                                           wavelength_0=wvln_wfs, 
                                           q=6)
    
    # pass the wavefront through the pyramid optic. 
    wavefront = pwfs.forward(wavefront)
    
    # camera = hp.NoiselessDetector(pwfs_grid)
    # camera.integrate(wf, 1)
    # image_ref = camera.read_out()
    # image_ref /= image_ref.sum()
    
    # plotter.plot_WFS_signal(wavefront)
    
    
    # %% In[Recover Wavefront Slopes]
    # =========================================================================
    # Recover wavefront slopes
    # =========================================================================
    WFS_signal = wavefront.intensity
    
    plotter.plot_light_progression(
        aperture, 
        incoming_wavefront.phase.shaped, 
        focal_plane.intensity.shaped, 
        pwfs.pyramid.phase(wvln_wfs).shaped,
        WFS_signal.shaped
        )
    
    
    # %%
    points = wavefront.grid.points
    X, Y = points.T
    # Create a boolean mask for the points in each quadrant
    Q1_mask = (X>0) & (Y>0)
    Q2_mask = (X<0) & (Y>0)
    Q3_mask = (X<0) & (Y<0)
    Q4_mask = (X>0) & (Y<0)
    # Extract the points in the quadrant
    Q1 = np.array([X[Q1_mask], Y[Q1_mask]]).T
    
    # %%
    Qs = []
    for mask in [Q1_mask, Q2_mask, Q3_mask, Q4_mask]:
        # Extract the image of the quadrant
        Q = WFS_signal[mask]
        Q = np.reshape(Q, [N_pupil_px_WFS, N_pupil_px_WFS])
        Qs.append(Q)
        
    
    
    
    quad_cell = np.array([[Qs[1], Qs[2]], 
                          [Qs[0], Qs[3]]])
    
    # Plot the extracted quadrants
    plotter.show_extracted(Qs)
    
    # Compute the WF slopes
    sx, sy = find_slopes(quad_cell)
    # sx *= N_pupil_px_WFS/telescope_dia
    # sy *= N_pupil_px_WFS/telescope_dia
    
    plotter.plot_2D_slopes(sx, sy)
    
    
    # %%
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    
    # # Plot the surface
    x, y = Q1.T
    x, y = x.reshape(sx.shape), y.reshape(sy.shape)
    plotter.plot_3D_slopes(x, y, sx, sy, './plots/recovered_slopes.html')
    
    
    
    
    # %%
    # Save the wavefront slopes as fits files
    hdu = fits.PrimaryHDU(sx)
    hdu.writeto('sx.fits', overwrite=True)
    hdu = fits.PrimaryHDU(sy)
    hdu.writeto('sy.fits', overwrite=True)
    
    # %%
    '''
    This code below works, but I would like to figure out how to improve the PyWFS 
    sampling. 
    
    # %% In[Use our own PyWFS]
    # =========================================================================
    # Now, lets make our own pyramid wavefront sensor optic, without using the
    # built in HCIpy one. 
    # =========================================================================
    # create a propagator for the light using a telecentric reimaging system
    # Propagator for pupil grid to focal grid
    prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)
    # propagator from focal grid to PyWFS grid
    prop_pywfs = hp.FraunhoferPropagator(focal_grid, pwfs_grid)
    
    
    # Create the incoming wavefront at the telescope aperture
    wavefront = hp.Wavefront(aperture.flatten(), wvln_wfs)
    wavefront.total_power = 1
    
    # Propagate the light to the telescope focal plane
    focal_plane = prop.forward(wavefront)
    
    # @TODO! How to find actual value for phase_delay instead of just guessing?
    pyramid = pyr.pyramid_phase_mask(focal_plane, N_pupil_px,
                                     phase_delay=6.86e-6, wvln=wvln_wfs)
    
    # plot the pyramid phase mask
    plotter.plot_pyramid_mask(pyramid)
    
    
    
    
    print(pyramid)
    focal_plane.electric_field *= np.exp(1j * pyramid.flatten())
    
    # Propagate the focal plane image to the pywfs grid
    pupil = prop_pywfs.forward(focal_plane)
    
    
    
    fp_img = np.log10(focal_plane.intensity/focal_plane.intensity.max())
    hp.imshow_field(fp_img, vmin=-6, cmap='bone')
    plt.colorbar()
    plt.show()
    
    hp.imshow_field(pupil.intensity, cmap='bone')
    plt.show()
    '''
    
    
    