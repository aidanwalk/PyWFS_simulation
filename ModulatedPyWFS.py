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
    def __init__(self, pupil=None, **kwargs):
        super().__init__(pupil, **kwargs)


    def modulate(self, wavefront, radius, num_steps, 
                 propagator=None, square_modulation=False):
        """
        Take a measurement of the wavefront using a modulated PyWFS. 

        Parameters
        ----------
        wavefront : hcipy.optics.wavefront.Wavefront
            HCIPy wavefront object representing the wavefront at the entrance
            of the telescope (i.e. in a pupil plane).
        radius : scalar
            Modulation radius in radians. 
        num_steps : int
            Number of modulation steps (how many points on the circle to 
            sample).
        propagator : callable, optional
            Propagator to use for the wavefront. If None, the default 
            propagator is the pyramid optic (i.e. the wavefront goes into four 
            pupil images). This can be used to propagate the wavefront to a 
            focal plane, if desired, like in visualize_modulation().
            The default is None.

        Returns
        -------
        signal : np.ndarray
            The intensity signal of the modulated PyWFS.
        """
                
        if square_modulation:
            # Generate discrete points along the outline of a square 
            # from vertex to vertex, the square should measure radius wide. 
            # This is why the modulation radius is divided by sqrt(2).
            modulation_positions = self.generate_square_modulation_positions(radius/2**0.5, num_steps)
        else:
            # Generate discrete points in a circle at which to steer the wavefront
            theta = np.linspace(0, 2*np.pi, num_steps, endpoint=False)
            x_modulation = radius * np.cos(theta)
            y_modulation = radius * np.sin(theta)
            modulation_positions = np.vstack((x_modulation.ravel(), y_modulation.ravel())).T
        
        # Pass the wavefront through the WFS at these points.  
        signal = self.discrete_modulation(wavefront, modulation_positions, propagator)

        return signal
    
    
    def generate_square_modulation_positions(self, radius, num_steps):
        """
        Generate modulation positions along the outline of a square.
        The square is rotated to match the orientation of the pyramid optic,
        such that square vertices land on the pyramid edges.

        parameters
        ----------
        radius : float
            The radius of the square (half the width and height).
        num_steps : int
            The number of points to generate along the square's perimeter.
        
        Returns
        -------
        modulation_positions : np.ndarray
            An array of shape (num_steps, 2) containing the x and y coordinates 
            of the modulation positions along the square's perimeter.
        """
        # The square should be radius wide and radius high, and contain num_steps points
        if num_steps < 4:
            raise ValueError("num_steps must be at least 4 to form a square outline")
        
        # Distribute points along the perimeter of the square
        # Each side gets approximately num_steps/4 points
        points_per_side = num_steps // 4
        remainder = num_steps % 4
        
        modulation_positions = []
        
        
        # Bottom side (left to right): y = -radius, x from -radius to radius
        x_bottom = np.linspace(-radius, radius, points_per_side + (1 if remainder > 0 else 0), endpoint=False)
        y_bottom = np.full_like(x_bottom, -radius)
        modulation_positions.extend(zip(x_bottom, y_bottom))
        
        # Right side (bottom to top): x = radius, y from -radius to radius
        y_right = np.linspace(-radius, radius, points_per_side + (1 if remainder > 1 else 0), endpoint=False)
        x_right = np.full_like(y_right, radius)
        modulation_positions.extend(zip(x_right, y_right))
        
        # Top side (right to left): y = radius, x from radius to -radius
        x_top = np.linspace(radius, -radius, points_per_side + (1 if remainder > 2 else 0), endpoint=False)
        y_top = np.full_like(x_top, radius)
        modulation_positions.extend(zip(x_top, y_top))
        
        # Left side (top to bottom): x = -radius, y from radius to -radius
        y_left = np.linspace(radius, -radius, points_per_side + (1 if remainder > 3 else 0), endpoint=False)
        x_left = np.full_like(y_left, -radius)
        modulation_positions.extend(zip(x_left, y_left))
        
        modulation_positions = np.array(modulation_positions)
        
        
        # If the pyramid array optic is rotated, the modulation points are 
        # oriented correctly. If not, we need to rotate them by 45 degrees.
        if hasattr(self.pyramidOptic, 'rotated') and self.pyramidOptic.rotated:
            return modulation_positions
        
        else:
            rot_mat = np.array(
                [[np.cos(np.pi/4), -np.sin(np.pi/4)],
                [np.sin(np.pi/4), np.cos(np.pi/4)]]
            )

            return modulation_positions @ rot_mat.T


    def discrete_modulation(self, wavefront, modulation_positions, propagator=None):
        """
        Passes the wavefront through discrete positions in the focal plane, 
        and integrates the light over each position. 

        Parameters
        ----------
        wavefront : hcipy.optics.wavefront.Wavefront
            HCIPy wavefront object representing the wavefront at the entrance
            of the telescope (i.e. in a pupil plane).
        modulation_positions : np.ndarray
            Focal plane positions to place the wavefront. Positions are in 
            units of radians. 
        propagator : callable, optional
            Propagator to use for the wavefront. If None, the default 
            propagator is the pyramid optic (i.e. the wavefront goes into four 
            pupil images). This can be used to propagate the wavefront to a 
            focal plane, if desired, like in visualize_modulation().
            The default is None.
        
        Returns
        -------
        signal : np.ndarray
            The intensity signal of the modulated PyWFS.

        """
        if propagator is None:
            propagator = self.pass_through

        # Create a tip-tilt mirror to steer the wavefront
        tip_tilt_mirror = hp.optics.TipTiltMirror(wavefront.grid)

        # Compute the wavefront after passing through the wavefront sensor 
        # at each modulation position. 
        signal = np.zeros(propagator(wavefront).shape)
        for point in modulation_positions:
            tip_tilt_mirror.actuators = point/2
            modulated_wavefront = tip_tilt_mirror.forward(wavefront)
            
            signal += propagator(modulated_wavefront)
        
        return signal




    def visualize_modulation(self, wavefront, radius, num_steps):
        pupil_signal = self.modulate(wavefront, radius, num_steps)
        focal_signal = self.modulate(wavefront, radius, num_steps, 
                                     propagator=self.pupil_to_focal_intensity)
        return focal_signal, pupil_signal
    

    def visualize_discrete_modulation(self, wavefront, modulation_positions):
        """
        Visualize the modulation of the wavefront sensor. 

        Parameters
        ----------
        wavefront : hcipy.optics.wavefront.Wavefront
            HCIPy wavefront object representing the wavefront at the entrance
            of the telescope (i.e. in a pupil plane).
        modulation_positions : np.ndarray
            Focal plane positions to place the wavefront. Positions are in 
            units of radians. 

        Returns
        -------
        signal : np.ndarray
            The intensity signal of the modulated PyWFS.
        """
        pupil_signal = self.discrete_modulation(wavefront, modulation_positions)
        focal_signal = self.discrete_modulation(wavefront, modulation_positions, 
                                                propagator=self.pupil_to_focal_intensity)
        return focal_signal, pupil_signal