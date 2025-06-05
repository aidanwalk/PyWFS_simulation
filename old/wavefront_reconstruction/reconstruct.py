#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example wavefront reconstruction utilizing the code recieved from Suzzanne. 
This is for Southwell geometry. 

Created on Tue Feb 25 11:22:35 2025

@author: Aidan Walk
"""

import southwellGeometry1 as sw

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

import sys
path2code = '/home/arcadia/mysoft/gradschool/useful/'
sys.path.append(path2code)
import code_fragments.Wavefront as wf # type: ignore
from code_fragments import Zernike # type: ignore


path2sim = '/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/'
sys.path.append(path2sim)
import plotter
from PyWFS import WaveFrontSensor


def plot_phase(x, y, p, fname='recovered_phase.html'):
    # HTML 3D Slopes Plot
    fig = make_subplots(rows=1, cols=1, 
                        specs=[[{'type':'scene'}]], 
                        subplot_titles=['X Slope'])
    # Create a plot for the x-slope
    trace_x = go.Surface(x=x, y=y, z=p)
    
    # Define the scene of the plot
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='slope',
        aspectratio= {"x": 1, "y": 1, "z": 0.5})
    
    # Create the figure
    # fig = go.Figure(data=[trace_x, trace_y], layout=layout)
    fig.add_trace(trace_x, row=1, col=1)
    fig.update_scenes(scene)

    fig.write_html(fname)
    plt.clf()
    
    
    return

if __name__ == "__main__":
    N_pupil_px = 2**8
    WFE = np.radians(0.01/3600)
    
    # Create the telescope aperture
    pupil_array = wf.circular_aperture((N_pupil_px,N_pupil_px),
                                       N_pupil_px/2)
    
    # Init the wavefront sensor
    WFS = WaveFrontSensor(pupil_array)
    
    
    # Create a wavefont incoming to the WFS
    incoming_wavefront = WFS.flat_wavefront()
    # Inject an aberration in to the incoming wavefront
    Z = Zernike.Zernike(pupil_array, rmax=N_pupil_px/2, wvln=WFS.wavelength)
    aberration = Z.Tilt_X(WFE=WFE, wvln=WFS.wavelength)
    incoming_wavefront.electric_field *= np.exp(1j * aberration.flatten())
    
    
    # Pass the wavefront through the WFS
    signal = WFS.pass_through(incoming_wavefront)
    # Recover the slopes
    sx, sy = WFS.measure_slopes(signal)
    sx = sx.T
    sy = sy.T

    s = np.hstack((sx.ravel(), sy.ravel()))
    

    # Reconstruct the wavefront
    A = sw.southwell1(sx.shape[0])
    M = np.linalg.pinv(A)
    
    
    p = M@s
    
    p = p.reshape(sx.shape)
    
    
    x = np.arange(sx.shape[0])
    x, y = np.meshgrid(x,x)
    plot_phase(x, y, p)
    
    plt.figure(figsize=(6,5))
    plt.title('Reconstructed Wavefront -- Suzzanne')
    im = plt.imshow(p.T, origin='lower')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('reconstructed_wavefront_Suzzanne.png', dpi=300)
