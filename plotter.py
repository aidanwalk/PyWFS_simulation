#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:09:46 2025

@author: Aidan Walk
"""

from old.pyramidWFS_hcipy import *

import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import hcipy as hp


def plot_pyramid_mask(pyramid):
    plt.figure('Pyramid Phase Mask')
    plt.clf()
    
    plt.title('Pyramid Phase Mask')
    im = plt.imshow(pyramid, origin='lower')
    plt.colorbar(im)
    
    plt.show()
    
    return
    


def plot_focal_plane(focal_plane):
    plt.figure('Focal Plane')
    plt.title('Focal Plane Image')
    fp_img = np.log10(focal_plane.intensity/focal_plane.intensity.max())
    hp.imshow_field(fp_img, vmin=-6, cmap='bone')
    plt.xlabel('x [arcsec]')
    plt.ylabel('y [arcsec]')
    ticks = np.linspace(-focal_extent/2, focal_extent/2, 5)
    labels = np.linspace(-focal_extent/2*206265, focal_extent/2*206265, 5)
    plt.xticks(ticks=ticks, 
               labels=np.round(labels,2))
    plt.yticks(ticks=ticks, 
               labels=np.round(labels,2))
    plt.colorbar(label='log(I)')
    plt.show()
    
    return


def plot_telescope_aperture(aperture):
    plt.figure('Aperture')
    plt.title('Telescope Aperture')
    hp.imshow_field(aperture.flatten(), cmap='bone')
    plt.colorbar(label='transmission')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.savefig('./plots/telescope_aperture.png', dpi=300)
    plt.clf()
    return 


def plot_wavefront_phase(phase):
    
    plt.figure('Wavefront Phase', figsize=(6,5))
    plt.clf()
    plt.title('Incoming Wavefront Phase')
    im = plt.imshow(phase, cmap='hsv', origin='lower')
    # Add a color bar which maps values to colors.
    plt.colorbar(im, fraction=0.046, pad=0.04, label='radians')
    
    plt.tight_layout()
    plt.savefig('./plots/incoming_wavefront.png', dpi=300)
    plt.show()
    
    
    return


def plot_WFS_signal(wavefront):
    plt.figure("WFS")
    plt.title('PyWFS Intensity')
    hp.imshow_field(wavefront.intensity, cmap='bone')
    plt.colorbar()
    plt.show()
    
    return


def plot_light_progression(aperture, 
                           aberration, 
                           focal_plane, 
                           pyramid, 
                           WFS_signal, 
                           fname='./plots/light_progression.png'):
    
    fig, ax = plt.subplots(nrows=1, ncols=4, 
                           tight_layout=True, 
                           figsize=(12,3))
    plt.suptitle('WFS Light Progression')
    
    # First, plot the incoming wavefront aberration
    ax[0].set_title('Incoming Wavefront Phase')
    im = ax[0].imshow(aberration, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    # Overplot the aperture of the telescope
    alpha = ~aperture
    ax[0].imshow(~aperture, alpha=alpha.astype(float), cmap='Greys')
    ax[0].axis('off')
    
    
    ax[1].set_title('Focal Plane PSF')
    img = np.log10(focal_plane / focal_plane.max())
    im = ax[1].imshow(img, cmap='bone', vmin=-6, vmax=0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax[1].axis('off')
    
    
    ax[2].set_title('Pyramid Phase Mask')
    im = ax[2].imshow(pyramid, cmap='hsv')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax[2].axis('off')
    
    ax[3].set_title('WFS Signal')
    im = ax[3].imshow(WFS_signal, cmap='bone')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax[3].axis('off')
    
    
    plt.savefig(fname, dpi=300)
    
    
    return 


def plot_2D_slopes(sx, sy, fname='./plots/recovered_WFS_slopes_2D.png'):
    plt.figure(tight_layout=True, figsize=(6,3))
    plt.suptitle("Recovered Wavefront Sensor Slopes")
    
    # Plot x slopes
    plt.subplot(121)
    plt.title('$s_{x}$')
    vmin, vmax = np.nanmin(sx), np.nanmax(sx)
    im = plt.imshow(sx, vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # plot y slopes
    plt.subplot(122)
    plt.title('$s_{y}$')
    im = plt.imshow(sy, vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.savefig(fname, dpi=300)
    plt.clf()
    return


def plot_3D_slopes(x, y, sx, sy, fname='recovered_slopes.html'):
    
    # plt.figure('slopes')
    # plt.quiver(x, y, sx, sy)
    # # plt.aspect('equal')
    # plt.show()
    
    # HTML 3D Slopes Plot
    fig = make_subplots(rows=1, cols=2, 
                        specs=[[{'type':'scene'},{'type':'scene'}]], 
                        subplot_titles=['X Slope', 'Y Slope'])
    # Create a plot for the x-slope
    trace_x = go.Surface(x=x, y=y, z=sx)
    trace_y = go.Surface(x=x, y=y, z=sy)
    
    # Define the scene of the plot
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='slope',
        aspectratio= {"x": 1, "y": 1, "z": 0.5})
    
    # Create the figure
    # fig = go.Figure(data=[trace_x, trace_y], layout=layout)
    fig.add_trace(trace_x, row=1, col=1)
    fig.add_trace(trace_y, row=1, col=2)
    fig.update_scenes(scene)

    fig.write_html(fname)
    plt.clf()
    
    
    return
    
    
def plot_phase(x, y, p, fname='./plots/recovered_phase.html'):
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
        zaxis_title='phase',
        aspectratio= {"x": 1, "y": 1, "z": 0.5})
    
    # Create the figure
    # fig = go.Figure(data=[trace_x, trace_y], layout=layout)
    fig.add_trace(trace_x, row=1, col=1)
    fig.update_scenes(scene)

    fig.write_html(fname)
    plt.clf()
    
    
    return
    
    
def show_extracted(Qs):
    
    plt.figure(figsize=(6,6))
    plt.clf()
    plt.suptitle('Quadrant Cutouts')
    
    plt.subplot(221)
    plt.title('Q2')
    plt.imshow(Qs[1], origin='lower', cmap='bone')
    plt.axis('off')
    
    plt.subplot(222)
    plt.title('Q1')
    plt.imshow(Qs[0], origin='lower', cmap='bone')
    plt.axis('off')
    
    plt.subplot(223)
    plt.title('Q3')
    plt.imshow(Qs[2], origin='lower', cmap='bone')
    plt.axis('off')
    
    plt.subplot(224)
    plt.title('Q4')
    plt.imshow(Qs[3], origin='lower', cmap='bone')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return
    