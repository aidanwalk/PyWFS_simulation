#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

A Wavefront Sensor class for a Pyramid Wavefront Sensor. 
--------------------------------------------------------
This is a wave optics simulator utilizing HCIPy and Fraunhofer diffraction
theory. 


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
        self.pyramidOptic = hp.PyramidWavefrontSensorOptics(
            self.pwfs_grid, 
            separation=self.telescope_diameter, 
            wavelength_0=self.wavelength, 
            q=6
            )
        self.pupil2pupils = self.pyramidOptic.forward
        
        
        
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
        """
        Creates a flat wavefront at the telescope aperture. The wavefront has 
        total power = 1. 

        Returns
        -------
        wavefront : hcipy.optics.wavefront.Wavefront
            A flat HCIPy wavefront (constant phase).

        """
        wavefront = hp.Wavefront(self.aperture.flatten(), self.wavelength)
        wavefront.total_power = 1
        return wavefront
    
    
    def pass_through(self, wavefront):
        """
        Passes a wavefront through the wavefront sensor. The input wavefront 
        should be the wavefront incident at the telescope aperture (i.e. in 
        pupil plane). 

        Parameters
        ----------
        wavefront : hcipy.optics.wavefront.Wavefront
            HCIPy wavefront object representing the wavefront at the entrance
            of the telescope (i.e. in a pupil plane).

        Returns
        -------
        pupil_images : list of ndarrays
            Wavefront sensor pupil images ordered by quadrant.

        """
        
        # Pass the incoming wavefront through the PyWFS
        WFS_signal = self.pupil2pupils(wavefront)
        # Split the WFS signal into its four quadrants (one pupil image for 
        # each pyramid facet)
        pupil_images = self.split_quadrants(WFS_signal.intensity, 
                                            WFS_signal.grid.points)

        return pupil_images
    


    def modulate(self, wavefront, radius, num_steps):
        # This is a modified version of ModulatedPyramidWavefrontSensorOptics
        # in hcipy.wavefront_sensing.pyramid

        # Create a tip-tilt mirror to steer the wavefront
        tip_tilt_mirror = hp.optics.TipTiltMirror(wavefront.grid)

        theta = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)
        x_modulation = radius / 2 * np.cos(theta)
        y_modulation = radius / 2 * np.sin(theta)
        

        modulation_positions = hp.field.CartesianGrid(hp.field.UnstructuredCoords((x_modulation, y_modulation)))
        
        wf_modulated = []
        for point in modulation_positions.points:
            tip_tilt_mirror.actuators = point
            modulated_wavefront = tip_tilt_mirror.forward(wavefront)
            
            wf_modulated.append(self.pyramidOptic.forward(modulated_wavefront))

        signal = np.zeros(wf_modulated[0].intensity.shape)
        for wf in wf_modulated:
            signal += wf.intensity


        return self.split_quadrants(signal, wf_modulated[0].grid.points)
    


    def split_quadrants(self, WFS_signal, points):
        """
        Splits the WFS signal (four pupil images in a single array) into 
        Four individual arrays (one array per pupil image)

        Parameters
        ----------
        wavefront : hcipy.optics.wavefront.Wavefront
            HCIPy wavefront object representing the wavefront at the pupil
            plane of the WFS.

        Returns
        -------
        Qs : list of ndarrays
            Wavefront sensor pupil images ordered by quadrant.

        """
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
        """
        Measures the wavefront slopes in <x> and <y> based on the input 
        pupil images (i.e. quadrants). 

        Parameters
        ----------
        quadrants : list of ndarrays
            A list of the WFS pupil images. The list must be of length 4 -- One 
            list item per pupil image. The list should be ordered [quadrant_1, 
            quadrant_2, quadrant_3, quadrant_4]

        Returns
        -------
        sx : ndarray
            Wavefront sensor slopes in <x>.
        sy : ndarray
            Wavefront sensor slopes in <y>.

        """
        # Construct the quad-cell 
        a,c,d,b = quadrants
        # Compute the mean intensity per pixel
        I = a+b+c+d
        
        # Compute the WFS slopes based on a quad-cell between the four pupil 
        # images
        sx = ((a+b)-(c+d)) / I
        sy = ((a+c)-(b+d)) / I
        
        
        sx *= self.telescope_diameter / self.N_elements
        sy *= self.telescope_diameter / self.N_elements
        
        return sx, sy
        
        
    def light_progression(self, wavefront):
        """
        Generates images of the wavefront progression through the PyWFS.

        Parameters
        ----------
        wavefront : hcipy.optics.wavefront.Wavefront
            HCIPy wavefront object representing the wavefront at the entrance
            of the telescope (i.e. in a pupil plane).

        Returns
        -------
        images : list of ndarrays
            List of images showing the wavefront progression through the WFS.

        """
        pupil_image = wavefront.phase.shaped
        focal_image = self.pupil2image(wavefront).intensity.shaped
        pyramid_image = self.pyramidOptic.pyramid.phase(self.wavelength).shaped
        WFS_signal = self.pupil2pupils(wavefront).intensity.shaped

        return pupil_image, focal_image, pyramid_image, WFS_signal
