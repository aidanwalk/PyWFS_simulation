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


path2sim = '/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/'
sys.path.append(path2sim)
import plotter
from PyWFS import WavefrontSensor
import aberrations


if __name__ == "__main__":
    N_pupil_px = 2**8
    WFE = np.radians(0.1/3600)
    
    # Init the wavefront sensor
    WFS = WavefrontSensor()
    
    
    # Create a wavefont incoming to the WFS
    incoming_wavefront = WFS.flat_wavefront()
    # Inject an aberration in to the incoming wavefront
    Z = aberrations.Zernike(WFS.input_pupil_grid, WFS.telescope_diameter)
    aberration = Z.from_name('defocus', WFE=WFE, wavelength=WFS.wavelength)
    incoming_wavefront = aberrations.aberrate(incoming_wavefront, aberration)
    
    
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
    