#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 20:04 2025

@author: Aidan Walk
"""



import re
import numpy as np
import hcipy as hp
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import cm
plt.close('all')



def plot_progression(focal_image, pupil_image, 
                     title='Modulated Pyramid',
                     fname='light_progression.png'):
    plt.figure(figsize=(7,4.2), tight_layout=True)
    plt.suptitle(title)

    plt.subplot(121)
    plt.title('focal plane')
    # im = plt.imshow(np.log10(focal_image/focal_image.max()), 
    #                 vmin=-5, vmax=0, cmap='bone', origin='lower')
    hp.imshow_field(np.log10(focal_image/focal_image.max()),
                    grid_units=1/206265,
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
    return 



def plot_response(data_file,
                  title='PyWFS Gain',
                  fname='response.png'):
    plt.figure(figsize=(8,6), tight_layout=True)
    plt.clf()

    
    tab = Table.read(data_file, format='ascii.fixed_width')
    
    
    colors = cm.inferno(np.linspace(0, 1, len(tab.columns[:])))
    for i, curve in enumerate(tab.columns[1:].keys()):
        radius = float(re.sub('[^0-9,.]','', curve))
        plt.plot(tab['input_WFE'], tab[curve], color=colors[i],
                 label='$r_{mod}$='+f'{radius:0.2f} as')
    
    plt.legend()
    # plt.axis('equal')
    plt.title(title)
    plt.xlabel('input wavefront slope [arcsec]')
    plt.ylabel('output wavefront slope [radians?]')
    plt.ylim(0.0, 0.06)
    plt.savefig(fname, dpi=300)
    return



def plot_phases(input_phases, output_phases, fname='phase_comparison.html', title="input vs recovered phase"):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    plt.suptitle(title)
    for i in range(len(input_phases)):
        input_phase = input_phases[i]
        recovered_phase = output_phases[i]
        # Create a plot for the x-slope
        pltkwargs = {'origin':'lower',
                    'cmap':'bone',
                    }
        im = axs[0,i].imshow(input_phase, **pltkwargs)
        plt.colorbar(im,fraction=0.046, pad=0.04)
        axs[0,i].axis('off')
        im = axs[1,i].imshow(recovered_phase, vmin=-recovered_phase.max(), 
                            vmax=recovered_phase.max(), 
                            **pltkwargs
                            )
        plt.colorbar(im,fraction=0.046, pad=0.04)
        axs[1,i].axis('off')
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    
    return