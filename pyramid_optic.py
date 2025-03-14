#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 08:34:15 2025

@author: Aidan Walk
"""

import numpy as np

import sys
path2code = '/home/arcadia/mysoft/gradschool/useful/'
sys.path.append(path2code)
from code_fragments import Zernike
from code_fragments import Coordinates as coords

def facet_mask(pts):
    # Facets of the pyramid
    Q1 = (pts[0]>=0) & (pts[1]>0)
    Q2 = (pts[0]<0) & (pts[1]>=0)
    Q3 = (pts[0]<=0) & (pts[1]<0)
    Q4 = (pts[0]>0) & (pts[1]<=0)
    
    return [Q1, Q2, Q3, Q4]


def edge_mask(pts):
    # Edges of the pyramid
    L1 = (pts[0] >0) & (pts[1]==0) # For the edge x>0, y=0
    L2 = (pts[0]==0) & (pts[1] >0) # For the edge y>0, x=0
    L3 = (pts[0] <0) & (pts[1]==0) # For the edge x<0, y=0
    L4 = (pts[0]==0) & (pts[1] <0) # For the edge y<0, x=0
    
    return [L1, L2, L3, L4]

def geometry_mask(grid):
    """
    Make a list of 4 arrays, where each array is a mask for each cartesian 
    quadrant. 

    Parameters
    ----------
    grid : TYPE
        DESCRIPTION.

    Returns
    -------
    mask : list
        list of array masks. 
        mask[0] = quadrant 1
        mask[1] = quadrant 2
        mask[2] = quadrant 3
        mask[3] = quadrant 4
        
    """
    # cartesian points cooresponding to the input grid
    pts = coords.coords(grid.shape, indexing='xy')
    
    # !TODO
    # This solution is not ideal since the edges of the pyramid are not
    # exactly between facets. This solution places the line y=0 in quadrant 1
    # That is, the line y=0 has a y-tilt. This is a sampling problem.
    # I attempted to fix this problem by adding the "edges". Edges only have 
    # tilt in one dimension, instead of two. However, this is a bad solution 
    # because you are essentially introducting another facet on the pyramid. 
    
    facets = facet_mask(pts)
    edges = edge_mask(pts)
    
    
    return facets, edges


def pyramid_phase_mask(focal_plane, N_pupil_px,
                       phase_delay=40, wvln=800e-9):
    """
    A Pyramid face simply induces a phase delay in the light equvalent to a 
    tip/tilt at the focal plane. We can make a "mask" for each pyramid face, 
    then apply a tip-tilt error in the focal plane image to emulate it passing 
    through the pyramid. 
    """
    
    
    Z = Zernike.Zernike(focal_plane.grid.ones().shaped,
                        rmax=N_pupil_px/2)
    
    facets, edges = geometry_mask(focal_plane.grid)
    
    wfe = phase_delay
    # Tilt each facet in the pyramid downward, with the point (0,0) anchored to
    # a height of 0. 
    facet1 = (Z.Tilt_X(-wfe, wvln=wvln)+Z.Tilt_Y(-wfe, wvln=wvln)) * facets[0]
    facet2 = (Z.Tilt_X( wfe, wvln=wvln)+Z.Tilt_Y(-wfe, wvln=wvln)) * facets[1]
    facet3 = (Z.Tilt_X( wfe, wvln=wvln)+Z.Tilt_Y( wfe, wvln=wvln)) * facets[2]
    facet4 = (Z.Tilt_X(-wfe, wvln=wvln)+Z.Tilt_Y( wfe, wvln=wvln)) * facets[3]
    
    # Tile each edge of the pyramid downward, with the point (0,0) anchored to 
    # a height of 0. 
    # edge1 = Z.Tilt_X(-wfe) * edges[0]
    # edge2 = Z.Tilt_Y()*-wfe * edges[1]
    # edge3 = Z.Tilt_X(wfe) * edges[2]
    # edge4 = Z.Tilt_Y()*wfe * edges[3]
    
    
    pyramid = facet1 + facet2 + facet3 + facet4
    # pyramid += edge1 + edge2 + edge3 + edge4
    
    
    return pyramid




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    mask = geometry_mask(np.ones([500,500]))[0]
    
    plt.figure()
    plt.clf()
    for i in range(len(mask)):
        plt.subplot(int(f'22{i+1}'))
        plt.title(f'Quadrant {i+1}')
        im = plt.imshow(mask[i], origin='lower')
        plt.colorbar(im)
        plt.axis('off')
    plt.show()
    