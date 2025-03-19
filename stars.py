#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:01 2025

@author: Aidan Walk
"""


import numpy as np


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

