#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Wavefront reconstruction using Southwell Geometry. 
--------------------------------------------------
The class interaction_matrix creates an interaction matrix object that allows
us to conventiently convert wavefront slopes in to phases (See method 
slope2phase). 

See ./examples/reconstruction/ for a simple example on how to use this class. 


Created on Tue Feb  4 09:31:32 2025

@author: Aidan Walk

"""

import numpy as np
import hcipy as hp
# from hcipy import inverse_tikhonov

class interaction_matrix(np.ndarray):
    """
    A synthetic interaction matrix for a Southwell geometry.
    This class is a subclass of numpy's ndarray and is used to create an
    interaction matrix for a Southwell geometry. The interaction matrix is
    used to convert wavefront slopes into phase values (see method 
    slope2phase).
    
    parameters
    ----------
    N : int
        The number of sub-apertures (elements) in one dimension. 
        The default is 36.
        
    """
    def __new__(cls, N=36, **kwargs):
        # Init the interaction matrix
        obj = np.empty((N**2,2*N**2)).view(cls)
        
        # Set the attributes
        obj.N = N
        # Populate the interaction matix
        obj[:] = obj.construct_interaction_matrix()
        
        return obj
        
    
    def construct_interaction_matrix(self):
        # Pupil plane sampling
        # Telescope Dia. in [m]
        self.A = self.construct_A() #/ self.h
        self.D = self.construct_D() / 2
        
        imat = np.linalg.pinv(self.A)@self.D
        # imat = inverse_tikhonov(self.A, rcond=1e-3, svd=None)@self.D
        
        return imat


    def construct_A(self):
        # The final array should be (dim**2 x 2*dim**2) in size
        A_x = self._make_sparse_x(-1, 1)
        # A_x = np.flipud(A_x)
        A_y = self._make_sparse_y(-1, 1)
        # A_y = np.flipud(A_y)
        # A = np.append(A, np.array([np.ones(A.shape[-1])]), axis=0)
        
        return np.vstack((A_x, A_y))
    
    
    def construct_D(self):
        # Performs the adjacent slope averaging
        # The final array should be (2*dim**2 x 2*dim**2) in size
        D_x = self._make_sparse_x(1,1)
        D_y = self._make_sparse_y(1,1)
        D_x = np.hstack((D_x, np.zeros(D_x.shape)))
        D_y = np.hstack((np.zeros(D_y.shape), D_y))
        
        # D = np.append(D, np.array([np.zeros((D.shape[0]))]).T, axis=1)
        
        return np.vstack((D_x, D_y))
    
    
    
    def _make_sparse_x(self, a, b):
        # construct the sparse matrix for x coefficients 
        # a is the coefficient on x[i,j]
        # b is the coefficient on x[i+1,j]
        # For example, for a 3x3 grid, this function will generate the matrix:
        #
        #   [a, b, 0, 0, 0, 0, 0, 0, 0]   |x1|
        #   [0, a, b, 0, 0, 0, 0, 0, 0]   |x2|
        #   [0, 0, 0, 0, 0, 0, 0, 0, 0]   |x3|
        #   [0, 0, 0, a, b, 0, 0, 0, 0]   |x4|
        #   [0, 0, 0, 0, a, b, 0, 0, 0] * |x5|
        #   [0, 0, 0, 0, 0, 0, 0, 0, 0]   |x6|
        #   [0, 0, 0, 0, 0, 0, a, b, 0]   |x7|
        #   [0, 0, 0, 0, 0, 0, 0, a, b]   |x8|
        #   [0, 0, 0, 0, 0, 0, 0, 0, 0]   |x9|
        #
        # For the grid: 
        # 
        #   [x1, x2, x3]
        #   [x4, x5, x6]
        #   [x7, x8, x9] - x-3,6,9 do not have a neighbor to the right of them 
        #                  (i.e. i+1 does not exist), so their coefficients 
        #                  become zero. 
        #
        
        A_x = np.zeros((self.N**2, self.N**2), dtype='int8')
        for i, row in enumerate(A_x):
            # If we are on the last coumn of the row, skip it, since this grid 
            # point does not have a neighbor to the right. 
            if i%self.N >= (self.N-1):
                continue
            # Otherwise, assign the correct values
            row[i] = a
            row[i+1] = b
            
        return A_x
    
    
    def _make_sparse_y(self, c1, c2):
        # construct the sparse matrix for y coefficients 
        # a is the coefficient on y[i,j]
        # b is the coefficient on y[i,j+1]
        # For example, for a 3x3 grid, this function will generate the matrix:
        #
        #   [a, 0, 0, b, 0, 0, 0, 0, 0]   |y1|
        #   [0, a, 0, 0, b, 0, 0, 0, 0]   |y2|
        #   [0, 0, a, 0, 0, b, 0, 0, 0]   |y3|
        #   [0, 0, 0, a, 0, 0, b, 0, 0]   |y4|
        #   [0, 0, 0, 0, a, 0, 0, b, 0] * |y5|
        #   [0, 0, 0, 0, 0, a, 0, 0, b]   |y6|
        #   [0, 0, 0, 0, 0, 0, 0, 0, 0]   |y7|
        #   [0, 0, 0, 0, 0, 0, 0, 0, 0]   |y8|
        #   [0, 0, 0, 0, 0, 0, 0, 0, 0]   |y9|
        #
        # For the grid: 
        # 
        #   [y1, y2, y3]
        #   [y4, y5, y6]
        #   [y7, y8, y9] - y-7,8,9 do not have a neighbor below them (i.e. j+1
        #                  does not exist), so their coefficients become zero. 
        #
        
        A_y = np.zeros((self.N**2, self.N**2), dtype='int8')
        for i, row in enumerate(A_y):
            # if this grid point does not have a neighbor below it, skip it. 
            # ((since each row is a flattened grid, the neighbor below a point 
            # is found at index (idx+dim), if idx+dim occurs beyond the last row
            # (i.e. at dim**2), it does not have a neighbor below it))
            if i+self.N >= self.N**2:
                continue
            # if i+dim < len(row):
            row[i] = c1
            row[i+self.N] = c2
            
        
        return A_y
    

    def slope2phase(self, sx, sy, telescope_dia=2.2, invert=True):
        # Construct the vector of slope values in the form:
        # [sx1, sx2, sx3, ..., sxf, sy1, sy2, sy3, ..., syf]
        s = np.hstack((sx.flatten(), sy.flatten()))
        
        # solve for the phase values using the instance as a matrix
        phases = np.array(self@s)
        
        # reshape the phases vector
        N = sx.shape[0]
        phases = phases.reshape((N,N))
        
        # ISSUE: For some reason my phases are mirrored about the X and Y axes. 
        # I am not sure why this happens... book keeping error in wavefront 
        # slopes calculation? 
        # SOLVED: The pyramid optic causes a flip in both the x and y axes. 
        # This causes the recovered phase to be upside down and mirrored.
        # (eg. if star light falls on the bottom right facet of the pyramid, 
        # the light will be reflected to the top left quadrant of the detector)
        # To fix this, we just flip the recovered phase.
        if invert:
            phases = phases[::-1, ::-1]

        return phases




class zernike_decomposition():
    """
    A class for Zernike decomposition of wavefront phase.
    """
    def __init__(self, N_modes, grid, D, starting_mode=1):
        self.starting_mode = starting_mode
        self.basis = hp.mode_basis.make_zernike_basis(N_modes, D, grid,
                                                      starting_mode=starting_mode)
        
        
        
    def decompose(self, phases):
        """
        Decomposes the wavefront phase into Zernike modes.
        
        Parameters
        ----------
        phases : array_like
            The wavefront phase to decompose.
        Returns
        -------
        coeffs : vector
            The coefficients on each Zernike mode.
        """
        return self.basis.coefficients_for(phases) # type: ignore
    
    
    def project(self, coeffs, **kwargs):
        """ 
        Projects a zernike decomposition back to a wavefront phase. That is, 
        what does the wavefront look like if we only keep the first N_modes
        
        """
        return self.basis.linear_combination(coeffs, **kwargs)
        

    # def get_noll_indicies(self):
    #     self.zernike_indices = hp.mode_basis.noll_to_zernike(self.starting_mode)



# .........................................................................
# The code below I wrote from scratch, it works well, but the version above 
# uses HCIPy and is more robust.
# .........................................................................

# import aberrations
# class zernike_decomposition(aberrations.Zernike):
#     """
#     A class for Zernike decomposition of wavefront phase.
#     """
#     def __init__(self, grid, D, **kwargs):
#         super().__init__(grid, D, **kwargs)
        
        
#     def decompose(self, phases, N_modes=10, **kwargs):
#         """
#         Decomposes the wavefront phase into Zernike modes.
        
#         Parameters
#         ----------
#         N_orders : int
#             The number of Zernike orders to decompose into.
        
#         Returns
#         -------
#         coeffs : vector
#             The coefficients on each Zernike mode.
#         """
#         # Get the Zernike modes
#         zernike_modes = self.get_zernike_modes(N_modes)
        
#         # Construct the Zernike basis set
#         Z = np.zeros((self.grid.shape[0]*self.grid.shape[1], N_modes))
#         for i, mode in enumerate(zernike_modes):
#             Z[:, i] = self.evaluate(*mode, **kwargs).flatten()
        
#         # Invert the Zernike basis set
#         Z_inv = np.linalg.pinv(Z)
#         # Get the coefficients
#         coeffs = Z_inv @ phases
#         return coeffs
    
    
#     def get_zernike_modes(self, N_modes=10):
#         """ 
#         Returns the (n,m) pairs of Zernike modes for a given number of orders.
#         (Noll's sequential indicies)
        
#         Parameters
#         ----------
#         N_orders : int
#             The first N Zernike orders.
#         """
#         assert len(self.COMMON_MODES) >= N_modes,\
#         f"Only {len(self.COMMON_MODES)}  Zernike modes available, attempting to get {N_modes}."
        
#         return list(self.COMMON_MODES.values())[0:N_modes]
        