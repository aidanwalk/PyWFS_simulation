#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:44:32 2024

@author: Aidan Walk
"""

import numpy as np
import matplotlib.pyplot as plt


import sys
path2code = '/home/arcadia/mysoft/gradschool/useful/'
sys.path.append(path2code)

from code_fragments import Wavefront as wf
from code_fragments import Zernike
from code_fragments import Coordinates as coords


def plot_wavefront(wavefront, **kwargs):
    plt.clf()
    im = plt.imshow(wavefront, origin='lower', **kwargs)
    plt.colorbar(im)
    plt.show()


if __name__ == "__main__":
    wvln = 500e-9
    WFE = 10e-3/206265
    # Create a circular aperture for incoming light
    wavefront = wf.circular_aperture(shape=(256, 256), r=256/2)
    
    # Create a wavefront at the aperature with some abberation
    Z = Zernike.Zernike(wavefront, rmax=256/2)
    prop = wf.propagator()
    wavefront = prop.abberation(wavefront, 
                                Z.Tilt_X(WFE)*wavefront, 
                                wvln=wvln)
    
    
    plot_wavefront(wf.phase(wavefront), cmap='hsv')
    
    
    # Image this wavefront into a focal plane
    # fp = wf.pupil_to_image(wavefront)
    # fp = np.fft.fftshift(fp)
    fp = prop.forward(wavefront, crop=None, wvln=wvln)
    
    
    
    im = wf.intensity(fp)
    plot_wavefront(np.log10(im / im.max()), vmin=-6, cmap='bone')
    
    # # Block out half the light from the focal plane
    # add a phase delay to half the light
    # Simulating a pyramid array. 
    """
    A Pyramid face simply induces a phase delay in the light equvalent to a 
    tip/tilt at the focal plane. We can make a "mask" for each pyramid face, 
    then apply a tip-tilt error in the focal plane image to emulate it passing 
    through the pyramid. 
    """
    
    mask1 = np.ones(fp.shape, dtype=bool)
    pts = coords.coords(mask1.shape)
    mask1[pts[0]>0] = False
    mask1[pts[1]>0] = False
    
    mask2 = np.ones(fp.shape, dtype=bool)
    mask2[pts[0]<0] = False
    mask2[pts[1]>0] = False
    
    mask3 = np.ones(fp.shape, dtype=bool)
    mask3[pts[0]<0] = False
    mask3[pts[1]<0] = False
    
    mask4 = np.ones(fp.shape, dtype=bool)
    mask4[pts[0]>0] = False
    mask4[pts[1]<0] = False
    
    wfe = 250e-5
    fp1 = prop.abberation(fp * mask1, Z.Tilt_X(-wfe)+Z.Tilt_Y()*-wfe)
    fp2 = prop.abberation(fp * mask2, Z.Tilt_X(wfe)+Z.Tilt_Y()*-wfe)
    fp3 = prop.abberation(fp * mask3, Z.Tilt_X(wfe)+Z.Tilt_Y()*wfe)
    fp4 = prop.abberation(fp * mask4, Z.Tilt_X(-wfe)+Z.Tilt_Y()*wfe)
    fp = fp1 + fp2 + fp3 + fp4
    
    
    im = wf.intensity(fp)
    plot_wavefront(np.log10(im / im.max()), vmin=-6, cmap='bone')
    
    
    # re image this light into a pupil plane
    # pupil = np.fft.ifftshift(fp)
    pupil_masked = prop.backward(fp, crop=256*2)
    im = wf.intensity(pupil_masked)
    # im = wf.phase(pupil_masked)
    # im = np.log10(im / im.max())
    plot_wavefront(im, cmap='bone')
    
    
    
    
    