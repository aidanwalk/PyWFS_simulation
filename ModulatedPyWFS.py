#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:01 2025

@author: Aidan Walk
"""

import hcipy as hp
import numpy as np
from PyWFS import WavefrontSensor


class ModulatedWavefrontSensor(WavefrontSensor):
    """
    A Modulated Pyramid Wavefront Sensor. 
    ---------------------------------------------------------------------------
    This is an extension of the PyWFS class to include modulation of the 
    incoming beam. 
    """
    def __init__(self, pupil, **kwargs):
        super().__init__(pupil, **kwargs)


    def modulate(self, wavefront, radius, num_steps, 
                 propagator=None):
        # This is a modified version of ModulatedPyramidWavefrontSensorOptics
        # in hcipy.wavefront_sensing.pyramid

        # Create a tip-tilt mirror to steer the wavefront
        tip_tilt_mirror = hp.optics.TipTiltMirror(wavefront.grid)

        theta = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)
        x_modulation = radius / 2 * np.cos(theta)
        y_modulation = radius / 2 * np.sin(theta)
        
        modulation_positions = np.vstack((x_modulation, y_modulation)).T
        

        if propagator is None:
            propagator = self.pupil2pupils

        # Compute the wavefront after passing through the wavefront sensor 
        # at each modulation position. 
        wf_modulated = []
        for point in modulation_positions:
            tip_tilt_mirror.actuators = point
            modulated_wavefront = tip_tilt_mirror.forward(wavefront)
            
            # wf_modulated.append(self.pyramidOptic.forward(modulated_wavefront))
            wf_modulated.append(propagator(modulated_wavefront))

        # Sum the modulated wavefront intensities to get the total modulated 
        # signal
        signal = np.zeros(wf_modulated[0].intensity.shaped.shape)
        for wf in wf_modulated:
            signal += wf.intensity.shaped

        return signal



    def visualize_modulation(self, wavefront, radius, num_steps):
        pupil_signal = self.modulate(wavefront, radius, num_steps)
        focal_signal = self.modulate(wavefront, radius, num_steps, 
                                     propagator=self.pupil2image)
        return focal_signal, pupil_signal