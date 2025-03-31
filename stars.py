#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:01 2025

@author: Aidan Walk
"""


import numpy as np



# -----------------------------------------------------------------------------
# FUNCTIONS TO GENERATE MODULATION POINTS
# -----------------------------------------------------------------------------

def uniform_azimuth(radius, N_points):
    # Generate discrete points in a circle at which to steer the wavefront
    theta = np.linspace(0, 2*np.pi, N_points, endpoint=False)
    x_modulation = radius * np.cos(theta)
    y_modulation = radius * np.sin(theta)
    
    modulation_positions = np.vstack((x_modulation, y_modulation)).T
    return modulation_positions


def random_azimuth(radius, N_points):
    # Generate random azimuthal angles for the modulation points
    theta = np.random.uniform(0, 2*np.pi, N_points)
    x_modulation = radius * np.cos(theta)
    y_modulation = radius * np.sin(theta)
    
    modulation_positions = np.vstack((x_modulation, y_modulation)).T
    return modulation_positions


def random_radius(radius, N_points):
    # Generate points randomly on the grid 
    return np.random.uniform(-radius, radius, (N_points,2))

# -----------------------------------------------------------------------------



def random_positions(extent, density):
    """
    Generate the x,y positions of stars randomly in a field. Coordinates (0,0)
    are centered in the field.  
    
    Parameters
    ----------
    extent : scalar
        The extent of the field in radians.
    density : scalar
        The mean density of stars in the field per steradian.
    """
    # Number of stars to generate
    N_stars = int(extent**2 * density)
    # Generate random positions
    x = np.random.uniform(-extent/2, extent/2, N_stars)
    y = np.random.uniform(-extent/2, extent/2, N_stars)

    points = np.vstack((x, y)).T
    return points

