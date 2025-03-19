#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:32:26 2025

@author: Aidan Walk
"""


import numpy as np
import hcipy as hp


class WavefrontSensor:
    """
    An unmodulated Pyramid Wavefront Sensor.
    ---------------------------------------------------------------------------
    This is a wave optics simulator utilizing HCIPy and Fraunhofer diffraction
    theory. 

    Parameters
    ----------
    pupil : ndarray
        A 2D array representing the telescope aperture. Values should be 
        between 0-1, representing the transparency at a given point. 
    wavelength : float, optional
        The wavelength of light in meters. The default is 800e-9.
    Npx_foc : int, optional
        The number of pixels across the focal plane. The default is 500.
    focal_extent : float, optional
        The extent of the focal plane in radians. The default is 5/206265.
    telescope_diameter : float, optional
        The diameter of the telescope in meters. The default is 2.2.
    N_elements : int, optional
        The total number of actuator elements in the pupil. The default is 36.
    """
    def __init__(self, 
                 pupil,
                 wavelength=800e-9, 
                 Npx_foc=500, 
                 focal_extent=5/206265,
                 telescope_diameter=2.2,
                 N_elements=36
                 ):
        
        # Assign attributes
        self.pupil = pupil                          
        self.Npx_pupil = self.pupil.shape[0]
        self.wavelength = wavelength                    # [meters]
        self.Npx_foc = Npx_foc
        self.focal_extent = focal_extent                # [radians]
        self.telescope_diameter = telescope_diameter    # [meters]
        self.N_elements = N_elements
        

        # Initialize the pupil, focal, and WFS grids
        self.__init_grids()
        
        # Create a Field at the aperture
        self.aperture = hp.Field(pupil, self.input_pupil_grid)
        
        # Create the propagator for go from pupil to focal grid
        self.pupil2image = hp.FraunhoferPropagator(self.input_pupil_grid, 
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
        self.input_pupil_grid = hp.make_pupil_grid(
            self.Npx_pupil, 
            self.telescope_diameter
            )
        # Make the pupil grid the for the output wavefront (four pupil images)
        # I am not entirely sure why this is one larger than pwfs_grid?
        self.output_pupil_grid = hp.make_pupil_grid(
            2*self.N_elements + 1, 
            2*self.telescope_diameter
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
        pupil_images : ndarray
            Intensity image of the WFS signal (an image of four pupils)

        """
        # Pass the incoming wavefront through the PyWFS
        wavefront = self.pupil2pupils(wavefront)
        return wavefront.intensity.shaped
    


    def split_quadrants(self, WFS_signal):
        """
        Splits the WFS signal (four pupil images in a single array) into 
        Four individual arrays (one array per pupil image)

        Parameters
        ----------
        WFS_signal : ndarray
            The intensity image of the wavefront sensor. (the array of four
            pupil images).

        Returns
        -------
        Qs : list of ndarrays
            Wavefront sensor pupil images ordered by quadrant.

        """
        WFS_signal = WFS_signal.ravel()
        assert WFS_signal.size == self.output_pupil_grid.x.size, \
            "The input signal shape does not match the output pupil grid."
        
        X, Y = self.output_pupil_grid.x, self.output_pupil_grid.y
        # Create a boolean mask for the points in each quadrant
        Q1_mask = (X>0) & (Y>0)
        Q2_mask = (X<0) & (Y>0)
        Q3_mask = (X<0) & (Y<0)
        Q4_mask = (X>0) & (Y<0)
        # Extract the points in the quadrant
        Qs = []
        for mask in [Q1_mask, Q2_mask, Q3_mask, Q4_mask]:
            # Extract the image of the quadrant
            Q = WFS_signal.ravel()[mask]
            Q = np.reshape(Q, [self.N_elements, self.N_elements])
            Qs.append(Q)
            
        return Qs
        
    
    
    def measure_slopes(self, WFS_signal):
        """
        Measures the wavefront slopes in <x> and <y> based on the input 
        pupil images (i.e. quadrants). 

        Parameters
        ----------
        WFS_signal : ndarray
            The intensity image of the wavefront sensor. (the array of four
            pupil images).

        Returns
        -------
        sx : ndarray
            Wavefront sensor slopes in <x>.
        sy : ndarray
            Wavefront sensor slopes in <y>.

        """
        # Construct the quad-cell 
        a,b,c,d = self.split_quadrants(WFS_signal)
        # Compute the mean intensity per pixel
        I = a+b+c+d
        
        # Compute the WFS slopes based on a quad-cell between the four pupil 
        # images
        sx = (a-b-c+d) / I
        sy = (a+b-c-d) / I
        
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
