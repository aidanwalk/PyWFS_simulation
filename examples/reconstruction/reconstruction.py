#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:50:44 2025

@author: Aidan Walk
"""

import sys
sys.path.append('/home/arcadia/mysoft/gradschool/699_1/simulation/PyWFS/')
from reconstruct import interaction_matrix



import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots



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
    # Open the slopes data
    sx = fits.getdata('sx.fits')
    sy = fits.getdata('sy.fits')
    
    # Create the interaction matrix
    imat = interaction_matrix(sx.shape[0])
    # Use it to solve for phases
    p = imat.slope2phase(sx, sy)
    
    # Make a plot of the recovered phase
    x = np.arange(p.shape[0])
    x, y = np.meshgrid(x, x)
    plot_phase(x, y, p, fname='recovered_phase.html')