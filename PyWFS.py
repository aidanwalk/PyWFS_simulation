#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:32:26 2025

@author: Aidan Walk
"""

import numpy as np
import hcipy as hp


class WaveFrontSensor:
    
    def __init__(self, pupil,
                 wavelength=800e-9, Npx_foc=500, focal_extent=5/206265,
                 telescope_diameter=2.2,
                 N_elements=36):
        
        self.pupil = pupil
        self.Npx_pupil = self.pupil.shape[0]
        self.wavelength = wavelength
        self.Npx_foc = Npx_foc
        self.focal_extent = focal_extent
        self.telescope_diameter = telescope_diameter
        self.N_elements = N_elements
        
        # Initialize the pupil, focal, and WFS grids
        self.__init_grids()
        
        # Create a Field at the aperture
        self.aperture = hp.Field(pupil, self.pupil_grid)
        
        # Create the propagator for go from pupil to focal grid
        self.pupil2image = hp.FraunhoferPropagator(self.pupil_grid, 
                                                   self.focal_grid).forward
        # And the propogator for the pyramid optic
        self.pupil2pupils = hp.PyramidWavefrontSensorOptics(
            self.pwfs_grid, 
            separation=self.telescope_diameter, 
            wavelength_0=self.wavelength, 
            q=6
            ).forward
        
        
        
    def __init_grids(self):
        # Initializes the pupil, focal, and WFS grids
        # Make the pupil grid (incoming pupil of the telescope)
        self.pupil_grid = hp.make_pupil_grid(
            self.Npx_pupil, 
            self.telescope_diameter
            )
        # Make the focal grid (This is where the pyramid array will be)
        self.focal_grid = hp.make_uniform_grid(
            [self.Npx_foc, self.Npx_foc], 
            [self.focal_extent, self.focal_extent]
            )
        # Make the grid for the wfs (grid the pupil images are projected on to)
        self.pwfs_grid = hp.make_pupil_grid(
            2*self.N_elements, 
            2*self.telescope_diameter
            )
        # Make a grid for only the first quadrant of the pyWFS (used for
        # plotting convenience)
        self.signal_grid = hp.make_pupil_grid(
            self.N_elements, 
            self.telescope_diameter
            )
        return 
        
        
    def flat_wavefront(self):
        # Returns a wavefront with no aberration, total power = 1
        wavefront = hp.Wavefront(self.aperture.flatten(), self.wavelength)
        wavefront.total_power = 1
        return wavefront
    
    
    def pass_through(self, wavefront):
        # Passes the wavefront through the wavefront sensor. Returns a list of
        # arrays, where each list item is a pupil image. 
        # index 0 = quadrant 1, index 1 = quadrant 2, etc.. 
        # Pass the incoming wavefront through the PyWFS
        WFS_signal = self.pupil2pupils(wavefront)
        # Split the WFS signal into its four quadrants (one pupil image for 
        # each pyramid facet)
        pupil_images = self.split_quadrants(WFS_signal)
        return pupil_images
    
    
    def split_quadrants(self, wavefront):
        # Split the WFS signal (four pupil images in a single array) into 
        # Four individual array (one array per pupil image)
        WFS_signal = wavefront.intensity
        points = wavefront.grid.points
        X, Y = points.T
        # Create a boolean mask for the points in each quadrant
        Q1_mask = (X>0) & (Y>0)
        Q2_mask = (X<0) & (Y>0)
        Q3_mask = (X<0) & (Y<0)
        Q4_mask = (X>0) & (Y<0)
        # Extract the points in the quadrant
        Qs = []
        for mask in [Q1_mask, Q2_mask, Q3_mask, Q4_mask]:
            # Extract the image of the quadrant
            Q = WFS_signal[mask]
            Q = np.reshape(Q, [self.N_elements, self.N_elements])
            Qs.append(Q)
            
        return Qs
        
    
    def measure_slopes(self, quadrants):
        # Construct the quad-cell 
        I = np.array([[quadrants[1], quadrants[2]], 
                      [quadrants[0], quadrants[3]]])
        
        # Compute the mean intensity per pixel
        I0 = (I[0,0]+I[0,1]+I[1,0]+I[1,1])/4
        
        # Compute the WFS slopes based on a quad-cell between the four pupil 
        # images
        sx = ( (I[0,1]+I[0,0]) - (I[1,1]+I[1,0]) ) / I0
        sy = ( (I[0,1]+I[1,1]) - (I[0,0]+I[1,0]) ) / I0
        
        sx *= self.telescope_diameter / self.N_elements
        sy *= self.telescope_diameter / self.N_elements
        return sx, sy
        
        
        
