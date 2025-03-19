#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Contains an example of how to use the PyWFS.py to simulate a WFS measurement. 

Created on Fri Mar 14 14:21:37 2025

@author: Aidan Walk
"""

import os
import sys
import numpy as np
from astropy.io import fits
# import matplotlib.pyplot as plt

path2code = '/home/arcadia/mysoft/gradschool/useful/code_fragments/'
sys.path.append(path2code)
import Wavefront as wf # type: ignore
import Zernike # type: ignore


path2sim = '/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/'
sys.path.append(path2sim)
import plotter
from PyWFS import WavefrontSensor


if __name__ == "__main__":
    N_pupil_px = 2**8
    WFE = np.radians(0.01/3600)
    
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = WavefrontSensor(pupil_array)
    
    
    # Create a wavefont incoming to the WFS
    incoming_wavefront = WFS.flat_wavefront()
    # Inject an aberration in to the incoming wavefront
    Z = Zernike.Zernike(pupil_array, rmax=N_pupil_px/2, wvln=WFS.wavelength)
    aberration = Z.Spherical(WFE=WFE, wvln=WFS.wavelength)
    incoming_wavefront.electric_field *= np.exp(1j * aberration.flatten())
    
    
    # Pass the wavefront through the WFS
    signal = WFS.pass_through(incoming_wavefront)
    # Recover the slopes
    sx, sy = WFS.measure_slopes(signal)
    
    
    x, y = WFS.signal_grid.points.T
    x, y = x.reshape(sx.shape), y.reshape(sy.shape)
    plotter.plot_3D_slopes(x, y, sx, sy, 'recovered_slopes_3D.html')
    plotter.plot_2D_slopes(sx, sy, 'recovered_slopes_2D.png')
    
    # Save the wavefront slopes as fits files
    hdu = fits.PrimaryHDU(sx)
    hdu.writeto('sx.fits', overwrite=True)
    hdu = fits.PrimaryHDU(sy)
    hdu.writeto('sy.fits', overwrite=True)
    