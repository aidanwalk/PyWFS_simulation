#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:18:03 2025

@author: Aidan Walk
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation



def iter_sw(sx, sy, N=36, telescope_dia=2.2, animate=False):
    sx = sx.T
    sy = sy.T
    # Initialize phases array
    p = np.zeros((N, N))
    # Initalize g_factors based on circular aperture
    g = construct_g((N,N), r=N/2)
    
    if animate:
        frames = []
    
    # Relaxation parameter to maximize rate of convergence
    w = 2 / (1+np.sin(np.pi/(N+1)))
    h = telescope_dia/N
    
    
    thresh = 0.5e-2
    converged = False
    
    c = 0
    while not converged:
        for j in range(p.shape[0]):
            for k in range(p.shape[1]):
                p1, p2, p3, p4 = get_p_ids(p, 
                                           (j+1,k), (j-1,k), 
                                           (j,k+1), (j,k-1), 
                                           fill=0)
                
                phi_bar = (p1 + p2 + p3 + p4) / g[j,k]
                
                
                sy1, sy2 = get_s_ids(sy, (j,k+1), (j,k-1))
                sx1, sx2 = get_s_ids(sx, (j+1,k), (j-1,k))
                bjk = (sy1 - sy2 + sx1 - sx2) * h/2
                
                bckt = phi_bar + bjk/g[j,k] - p[j,k]
                p[j,k] = p[j,k] + w * bckt
            
        if animate:
            frames.append(-p)
            
        c += 1
        if np.abs(bckt) < thresh or c>50:
            converged = True
            break
    
    if animate: make_animation(np.array(frames))
            
    return -p
    
    
     
def construct_g(shape, r):
    # interior
    grid = np.ones(shape) * 4
    # Edges
    grid[ :, 0] = 3
    grid[ :,-1] = 3
    grid[ 0, :] = 3
    grid[-1, :] = 3
    # Corners
    grid[ 0, 0] = 2
    grid[ 0,-1] = 2
    grid[-1, 0] = 2
    grid[-1,-1] = 2
    
    return grid
    
    
    
def get_p_ids(arr, *ids, fill=0):
    # arr = input array
    # ids = array indexes we want to extract **must be tuple of len(2)**
    # returns a list of len(ids), whose elements are the array elements 
    # cooresponding to ids
    vals = []
    for i,j in ids:
        # if the index exists, take the value at that index
        if (0 <= i < arr.shape[0]) and (0 <= j < arr.shape[1]):
            val = arr[i,j]
        # Otherwise, we must take the nagative of the adjacent value
        # For now, set this value to inf so we can find it later. 
        else: val = fill
        vals.append(val)
    return vals
        
    
    
def get_s_ids(arr, *ids):
    # Extract the values in arr at the indexes ids
    # if one of the indexes do not exist, take the negative of the adjacent 
    # index. This is possible becuase there should not exist a case where both
    # indexes do not exist. 
    vals = []
    
    
    for i,j in ids:
        # if the index exists, take the value at that index
        if (0 <= i < arr.shape[0]) and (0 <= j < arr.shape[1]):
            val = arr[i,j]
        # Otherwise, we must take the nagative of the adjacent value
        # For now, set this value to inf so we can find it later. 
        else: val = np.inf
        vals.append(val)
    
    
    # If one of the values does not exist, 
    # take the negative of the adjacent value
    vals = [-vals[i-1] 
            if vals[i]==np.inf
            else vals[i] 
            for i in range(len(vals))]
    
        
    return vals
        
    
    
def test_ids():
    # ** unit test for functions get_p_ids() and get_s_ids() **
    
    # Make an easy-to-read array for testing the functions
    A = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
    
    # -------------------------------------------------------------------------
    # Testing get_p_ids
    # -------------------------------------------------------------------------
    # When we pass an index outside of the dimensions of A, get_p_ids()
    # should return the fill value. Here, we expect to get a value of 0 since
    # the requested index does not exist. 
    a, = get_p_ids(A, (3,0), fill=0)
    assert a == 0, f'Incorrect fill value {a}. Expected 0'
    
    # Try to negative index the array. In this case, we should also get back
    # the fill value. 
    a, = get_p_ids(A, (-1,0), fill=0)
    assert a == 0, f'''Incorrect fill value {a}. Expected 0. 
    Negative index test failed'''
    
    # Make sure the function actually works. If we request index (2,0), we
    # expect 7. 
    a, = get_p_ids(A, (2,0), fill=0)
    assert a == A[2,0], 'Failed to return the requested index.'
    # -------------------------------------------------------------------------
    
    
    # -------------------------------------------------------------------------
    # Testing get_s_ids
    # -------------------------------------------------------------------------
    # When we pass an index outside of the dimensions of A, get_s_ids()
    # should return the negative of the adjacent index. Here, we expect to get 
    # a value of -7 since the requested index does not exist. 
    a,b = get_s_ids(A, (2,0), (3,0))
    assert (a == A[2,0]) and (b == -A[2,0]), f'Incorrect adjacent value {b}.'
    
    # Make sure the function actually works. If we request index (2,0) and 
    # (2,1), we expect 7 and 8. 
    a,b = get_s_ids(A, (2,0), (2,1))
    assert (a == 7) and (b==8), 'Failed to return the requested indices.'
    # -------------------------------------------------------------------------
    
    
    
def make_animation(frames, fname='animation.gif'):
    print('Making an animtion...')
    def update(i):
        im.set_data(frames[i])
        return im
    
    fig = plt.figure(tight_layout=True)
    im = plt.imshow(frames[0], 
                    vmin=frames[-1].min(), vmax=frames[-1].max(), 
                    cmap='hsv', 
                    origin='lower')
    plt.colorbar(im)
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=75)
    ani.save(fname, writer='pillow')
    print('Animation saved to:', fname)
    return
    

if __name__ == "__main__":
    from astropy.io import fits
    animate=True
    # Open the slopes data
    sx = fits.getdata('sx.fits')
    sy = fits.getdata('sy.fits')
    
    phase = iter_sw(sx, sy, N=sx.shape[-1], telescope_dia=2.2, animate=animate)
    
    