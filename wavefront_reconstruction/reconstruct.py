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
    # open the slopes arrays
    sx = fits.getdata('../sx.fits').T
    sy = fits.getdata('../sy.fits').T
    s = np.hstack((sx.ravel(), sy.ravel()))
    
    A = sw.southwell1(sx.shape[0])
    M = np.linalg.pinv(A)
    
    
    p = M@s
    
    p = p.reshape(sx.shape)
    
    
    x = np.arange(sx.shape[0])
    x, y = np.meshgrid(x,x)
    plot_phase(x, y, p)
    
    plt.figure(figsize=(6,5))
    plt.title('Reconstructed Wavefront -- Suzzanne')
    im = plt.imshow(p.T, cmap='hsv', origin='lower')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('reconstructed_wavefront_Suzzanne.png', dpi=300)
    plt.show()
