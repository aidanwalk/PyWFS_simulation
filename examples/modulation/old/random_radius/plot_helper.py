#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 20:04 2025

@author: Aidan Walk
"""




import numpy as np
import hcipy as hp
from astropy.io import fits
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



def plot_response(input_slopes, curves, modulation_radii, 
                  title='PyWFS Gain',
                  fname='response.png'):
    plt.figure(figsize=(8,6), tight_layout=True)
    plt.clf()

    colors = cm.inferno(np.linspace(0, 1, len(curves)+1))
    for i, curve in enumerate(curves):
        plt.plot(input_slopes, curve, color=colors[i])
    
    plt.legend()
    # plt.axis('equal')
    plt.title(title)
    plt.xlabel('input slopes [arcsec]')
    plt.ylabel('output slopes [?]')
    plt.ylim(-0.01, 0.06)
    plt.savefig(fname, dpi=300)
    return



def plot_phases(N_files, prefix='', fname='phase_comparison.html', title="input vs recovered phase"):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    plt.suptitle(title)
    for i in range(N_files):
        input_phase = fits.getdata(f'./aberrations/input_{prefix}_{i}.fits')
        recovered_phase = fits.getdata(f'./aberrations/recovered_{prefix}_{i}.fits')
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